# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import copy
import time
import json
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, matthews_corrcoef

from src.fp16 import network_to_half
from apex.fp16_utils import FP16_Optimizer

from ..utils import get_optimizer, concat_batches, truncate, to_cuda
from ..data.dataset import Dataset, ParallelDataset
from ..data.loader import load_binarized, set_dico_parameters


N_CLASSES = {
    'MNLI-m': 3,
    'MNLI-mm': 3,
    'QQP': 2,
    'QNLI': 2,
    'SST-2': 2,
    'CoLA': 2,
    'MRPC': 2,
    'RTE': 2,
    'STS-B': 1,
    'WNLI': 2,
    'AX': 3,
}


logger = getLogger()


class GLUE:

    def __init__(self, embedder, scores, params):
        """
        Initialize GLUE trainer / evaluator.
        Initial `embedder` should be on CPU to save memory.
        """
        self._embedder = embedder
        self.params = params
        self.scores = scores

    def get_iterator(self, splt):
        """
        Build dataOld iterator.
        """
        return self.data[splt]['x'].get_iterator(
            shuffle=(splt == 'train'),
            return_indices=True,
            group_by_size=self.params.group_by_size
        )

    def run(self, task):
        """
        Run GLUE training / evaluation.
        """
        params = self.params

        # task parameters
        self.task = task
        params.out_features = N_CLASSES[task]
        self.is_classif = task != 'STS-B'

        # load dataOld
        self.data = self.load_data(task)
        if not self.data['dico'] == self._embedder.dico:
            raise Exception(("Dictionary in evaluation dataOld (%i words) seems different than the one " +
                             "in the pretrained model (%i words). Please verify you used the same dictionary, " +
                             "and the same values for max_vocab and min_count.") % (len(self.data['dico']), len(self._embedder.dico)))

        # embedder
        self.embedder = copy.deepcopy(self._embedder)
        self.embedder.cuda()

        # projection layer
        self.proj = nn.Sequential(*[
            nn.Dropout(params.dropout),
            nn.Linear(self.embedder.out_dim, params.out_features)
        ]).cuda()

        # float16
        if params.fp16:
            assert torch.backends.cudnn.enabled
            self.embedder.model = network_to_half(self.embedder.model)
            self.proj = network_to_half(self.proj)

        # optimizer
        self.optimizer = get_optimizer(
            list(self.embedder.get_parameters(params.finetune_layers)) +
            list(self.proj.parameters()),
            params.optimizer
        )
        if params.fp16:
            self.optimizer = FP16_Optimizer(self.optimizer, dynamic_loss_scale=True)

        # train and evaluate the model
        for epoch in range(params.n_epochs):

            # update epoch
            self.epoch = epoch

            # training
            logger.info("GLUE - %s - Training epoch %i ..." % (task, epoch))
            self.train()

            # evaluation
            logger.info("GLUE - %s - Evaluating epoch %i ..." % (task, epoch))
            with torch.no_grad():
                scores = self.eval('valid')
                self.scores.update(scores)

    def train(self):
        """
        Finetune for one epoch on the training set.
        """
        params = self.params
        self.embedder.train()
        self.proj.train()

        # training variables
        losses = []
        ns = 0  # number of sentences
        nw = 0  # number of words
        t = time.time()

        iterator = self.get_iterator('train')
        lang_id = params.lang2id['en']

        while True:

            # batch
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if self.n_sent == 1:
                (x, lengths), idx = batch
                x, lengths = truncate(x, lengths, params.max_len, params.eos_index)
            else:
                (sent1, len1), (sent2, len2), idx = batch
                sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
                sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
                x, lengths, _, _ = concat_batches(sent1, len1, lang_id, sent2, len2, lang_id, params.pad_index, params.eos_index, reset_positions=False)
            y = self.data['train']['y'][idx]
            bs = len(lengths)

            # cuda
            x, y, lengths = to_cuda(x, y, lengths)

            # loss
            output = self.proj(self.embedder.get_embeddings(x, lengths, positions=None, langs=None))
            if self.is_classif:
                loss = F.cross_entropy(output, y, weight=self.weights)
            else:
                loss = F.mse_loss(output.squeeze(1), y.float())

            # backward / optimization
            self.optimizer.zero_grad()
            if params.fp16:
                self.optimizer.backward(loss)
            else:
                loss.backward()
            self.optimizer.step()

            # update statistics
            ns += bs
            nw += lengths.sum().item()
            losses.append(loss.item())

            # log
            if ns != 0 and ns % (10 * bs) < bs:
                logger.info(
                    "GLUE - %s - Epoch %s - Train iter %7i - %.1f words/s - %s Loss: %.4f"
                    % (self.task, self.epoch, ns, nw / (time.time() - t), 'XE' if self.is_classif else 'MSE', sum(losses) / len(losses))
                )
                nw, t = 0, time.time()
                losses = []

            # epoch size
            if params.epoch_size != -1 and ns >= params.epoch_size:
                break

    def eval(self, splt):
        """
        Evaluate on XNLI validation and test sets, for all languages.
        """
        params = self.params
        self.embedder.eval()
        self.proj.eval()

        scores = OrderedDict({'epoch': self.epoch})
        task = self.task.lower()

        pred = []  # predicted values
        gold = []  # real values

        lang_id = params.lang2id['en']

        for batch in self.get_iterator(splt):

            # batch
            if self.n_sent == 1:
                (x, lengths), idx = batch
                # x, lengths = truncate(x, lengths, params.max_len, params.eos_index)
            else:
                (sent1, len1), (sent2, len2), idx = batch
                # sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
                # sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
                x, lengths, _, _ = concat_batches(sent1, len1, lang_id, sent2, len2, lang_id, params.pad_index, params.eos_index, reset_positions=False)
            y = self.data[splt]['y'][idx]

            # cuda
            x, y, lengths = to_cuda(x, y, lengths)

            # prediction
            output = self.proj(self.embedder.get_embeddings(x, lengths, positions=None, langs=None))
            p = output.data.max(1)[1] if self.is_classif else output.squeeze(1)
            pred.append(p.cpu().numpy())
            gold.append(y.cpu().numpy())

        gold = np.concatenate(gold)
        pred = np.concatenate(pred)

        if self.is_classif:
            scores['%s_valid_acc' % task] = 100. * (pred == gold).sum() / len(pred)
            scores['%s_valid_f1' % task] = 100. * f1_score(gold, pred, average='binary' if params.out_features == 2 else 'micro')
            scores['%s_valid_mc' % task] = 100. * matthews_corrcoef(gold, pred)
        else:
            scores['%s_valid_prs' % task] = 100. * pearsonr(pred, gold)[0]
            scores['%s_valid_spr' % task] = 100. * spearmanr(pred, gold)[0]

        logger.info("__log__:%s" % json.dumps(scores))
        return scores

    def load_data(self, task):
        """
        Load pair regression/classification bi-sentence tasks
        """
        params = self.params
        data = {splt: {} for splt in ['train', 'valid', 'test']}
        dpath = os.path.join(params.data_path, 'eval', task)

        self.n_sent = 1 if task in ['SST-2', 'CoLA'] else 2

        for splt in ['train', 'valid', 'test']:

            # load dataOld and dictionary
            data1 = load_binarized(os.path.join(dpath, '%s.s1.pth' % splt), params)
            data2 = load_binarized(os.path.join(dpath, '%s.s2.pth' % splt), params) if self.n_sent == 2 else None
            data['dico'] = data.get('dico', data1['dico'])

            # set dictionary parameters
            set_dico_parameters(params, data, data1['dico'])
            if self.n_sent == 2:
                set_dico_parameters(params, data, data2['dico'])

            # create dataset
            if self.n_sent == 1:
                data[splt]['x'] = Dataset(data1['sentences'], data1['positions'], params)
            else:
                data[splt]['x'] = ParallelDataset(
                    data1['sentences'], data1['positions'],
                    data2['sentences'], data2['positions'],
                    params
                )

            # load labels
            if splt != 'test' or task in ['MRPC']:
                # read labels from file
                with open(os.path.join(dpath, '%s.label' % splt), 'r') as f:
                    lines = [l.rstrip() for l in f]
                # STS-B task
                if task == 'STS-B':
                    assert all(0 <= float(x) <= 5 for x in lines)
                    y = [float(l) for l in lines]
                # QQP
                elif task == 'QQP':
                    UNK_LABEL = 0
                    lab2id = {x: i for i, x in enumerate(sorted(set(lines) - set([''])))}
                    y = [lab2id.get(x, UNK_LABEL) for x in lines]
                # other tasks
                else:
                    lab2id = {x: i for i, x in enumerate(sorted(set(lines)))}
                    y = [lab2id[x] for x in lines]
                data[splt]['y'] = torch.LongTensor(y)
                assert len(data[splt]['x']) == len(data[splt]['y'])

        # compute weights for weighted training
        if task != 'STS-B' and params.weighted_training:
            weights = torch.FloatTensor([
                1.0 / (data['train']['y'] == i).sum().item()
                for i in range(len(lab2id))
            ]).cuda()
            self.weights = weights / weights.sum()
        else:
            self.weights = None

        return data
