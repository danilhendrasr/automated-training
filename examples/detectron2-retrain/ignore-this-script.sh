#!/bin/bash
# setup, download pretrained model, dataset prep
# setup
apt update && apt install -y python3-opencv
pip3 install opencv-python

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# download pretrained model
wget https://modelrepo-autotraining-poc-ndflx.s3.amazonaws.com/models/modelv1/model-metadata.json

wget https://modelrepo-autotraining-poc-ndflx.s3.amazonaws.com/models/modelv1/output-model.zip
unzip -q output-model.zip
rm output-model.zip

# dataset prep
wget https://modelrepo-autotraining-poc-ndflx.s3.amazonaws.com/datasets/vh-det-v01.zip

unzip -q vh-det-v01.zip
rm vh-det-v01.zip
