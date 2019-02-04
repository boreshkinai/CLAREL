FROM tensorflow/tensorflow:1.8.0-devel-gpu-py3

MAINTAINER Boris Oreshkin <boris@elementai.com>

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

RUN pip install tqdm gitpython googledrivedownloader requests pillow torch==0.4.1 keras==2.2.4 gensim==3.7.1
