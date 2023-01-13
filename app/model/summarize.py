# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     cat source_sentences.bpe | \
#     python translate.py --exp_name translate \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output
#


import torch

from model.src.utils import AttrDict
from model.src.utils import bool_flag, initialize_exp
from model.src.data.dictionary import Dictionary
from model.src.model.transformer import TransformerEncoder, TransformerDecoder

class Params:
    def __init__(self):
        self.beam_size = 1
        self.length_penalty = 1.0
        self.early_stopping = False

class Model:
    def __init__(self, model_path):
        self.__model = torch.load(model_path)
        self.__params = Params()
        self.__set_up_params()
        self.__initilize_encoder_decoder()
        
    def __set_up_params(self):
        self.__model_params = AttrDict(self.__model['params'])
        # update dictionary parameters
        for name in ['src_n_words', 'tgt_n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
            setattr(self.__params, name, getattr(self.__model_params, name))
    
    def __initilize_encoder_decoder(self):
        with torch.no_grad():
            self.__source_dico = Dictionary(self.__model['source_dico_id2word'], self.__model['source_dico_word2id'])
            self.__target_dico = Dictionary(self.__model['target_dico_id2word'], self.__model['target_dico_word2id'])
            self.__encoder = TransformerEncoder(self.__model_params, self.__source_dico, with_output=False).cuda().eval()
            self.__encoder.load_state_dict(self.__model['encoder'])
            self.__decoder = TransformerDecoder(self.__model_params, self.__target_dico, with_output=True).cuda().eval()
            self.__decoder.load_state_dict(self.__model['decoder'])
            
    def __describe(self, data, title):
        """valueLengths = []
        xLabelLengths = []
        yLabelLengths = []
        titleLengths = []"""
        
        enc_x1_ids = []
        enc_x2_ids = []
        enc_x3_ids = []
        enc_x4_ids = []
        for table_line, title_line in zip(data, title):
            record_seq = [each.split('|') for each in table_line.split()]
            assert all([len(x) == 4 for x in record_seq])

            enc_x1_ids.append(torch.LongTensor([self.__source_dico.index(x[0]) for x in record_seq]))
            enc_x2_ids.append(torch.LongTensor([self.__source_dico.index(x[1]) for x in record_seq]))
            enc_x3_ids.append(torch.LongTensor([self.__source_dico.index(x[2]) for x in record_seq]))
            enc_x4_ids.append(torch.LongTensor([self.__source_dico.index(x[3]) for x in record_seq]))

        enc_xlen = torch.LongTensor([len(x) + 2 for x in enc_x1_ids])
        enc_x1 = torch.LongTensor(enc_xlen.max().item(), enc_xlen.size(0)).fill_(self.__params.pad_index)
        enc_x1[0] = self.__params.eos_index
        enc_x2 = torch.LongTensor(enc_xlen.max().item(), enc_xlen.size(0)).fill_(self.__params.pad_index)
        enc_x2[0] = self.__params.eos_index
        enc_x3 = torch.LongTensor(enc_xlen.max().item(), enc_xlen.size(0)).fill_(self.__params.pad_index)
        enc_x3[0] = self.__params.eos_index
        enc_x4 = torch.LongTensor(enc_xlen.max().item(), enc_xlen.size(0)).fill_(self.__params.pad_index)
        enc_x4[0] = self.__params.eos_index

        for j, (s1,s2,s3,s4) in enumerate(zip(enc_x1_ids, enc_x2_ids, enc_x3_ids, enc_x4_ids)):
            if enc_xlen[j] > 2:  # if sentence not empty
                enc_x1[1:enc_xlen[j] - 1, j].copy_(s1)
                enc_x2[1:enc_xlen[j] - 1, j].copy_(s2)
                enc_x3[1:enc_xlen[j] - 1, j].copy_(s3)
                enc_x4[1:enc_xlen[j] - 1, j].copy_(s4)
            enc_x1[enc_xlen[j] - 1, j] = self.__params.eos_index
            enc_x2[enc_xlen[j] - 1, j] = self.__params.eos_index
            enc_x3[enc_xlen[j] - 1, j] = self.__params.eos_index
            enc_x4[enc_xlen[j] - 1, j] = self.__params.eos_index

        enc_x1 = enc_x1.cuda()
        enc_x2 = enc_x2.cuda()
        enc_x3 = enc_x3.cuda()
        enc_x4 = enc_x4.cuda()
        enc_xlen = enc_xlen.cuda()

        # encode source batch and translate it
        encoder_output = self.__encoder('fwd', x1=enc_x1, x2=enc_x2, x3=enc_x3, x4=enc_x4, lengths=enc_xlen)
        encoder_output = encoder_output.transpose(0, 1)

        max_len = 602
        if self.__params.beam_size <= 1:
            decoded, dec_lengths = self.__decoder.generate(encoder_output, enc_xlen, max_len=max_len)
        elif self.__params.beam_size > 1:
            decoded, dec_lengths = self.__decoder.generate_beam(encoder_output, enc_xlen, self.__params.beam_size, 
                                            self.__params.length_penalty, self.__params.early_stopping, max_len=max_len)
        
        output = []
        for j in range(decoded.size(1)):
            sent = decoded[:, j]
            delimiters = (sent == self.__params.eos_index).nonzero().view(-1)
            assert len(delimiters) >= 1 and delimiters[0].item() == 0
            sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]
            tokens = []
            for k in range(len(sent)):
                ids = sent[k].item()
                word = self.__target_dico[ids]
                tokens.append(word)
            target = " ".join(tokens)
            output.append(target)
        return output    

    def describe(self, data, title):
        with torch.no_grad():
            caption = self.__describe(data, title)
        return caption
    
