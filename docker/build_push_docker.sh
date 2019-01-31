#!/bin/bash

docker build -f Dockerfile -t zeroshot-pairwise-tensorflow-1.8.0 .
docker tag zeroshot-pairwise-tensorflow-1.8.0 images.borgy.elementai.lan/tensorflow/zeroshot-pairwise-tensorflow-1.8.0
docker push images.borgy.elementai.lan/tensorflow/zeroshot-pairwise-tensorflow-1.8.0
chmod +x ../train.py
