FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY requirements.txt requirements.txt


RUN conda install -c conda-forge spacy-model-en_core_web_md
RUN pip install --upgrade pip
RUN pip install -U scikit-learn
RUN pip install -U -r requirements.txt

COPY . /app
WORKDIR /app