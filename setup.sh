#!/usr/bin/env bash

conda install tqdm scikit-image

wget https://www.dropbox.com/s/l6w32zhoextlu2x/data.zip?dl=0
# apt-get install unzip
unzip data.zip?dl=0 > unzip_data.log

mkdir model_ckpt
