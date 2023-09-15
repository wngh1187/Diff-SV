FROM nvcr.io/nvidia/pytorch:21.04-py3

RUN pip3 install wrapt --upgrade --ignore-installed

RUN apt-get update 

RUN pip install torch==1.9.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
ADD requirements.txt .
RUN pip3 install -r requirements.txt

ENV PYTHONPATH /workspace/Diff-SV
WORKDIR /workspace/Diff-SV

RUN pip3 install wandb --upgrade
RUN pip3 install typing-extensions==4.3.0
RUN pip3 install awscli --ignore-installed six

