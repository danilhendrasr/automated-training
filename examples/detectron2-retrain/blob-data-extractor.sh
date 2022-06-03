#!/bin/bash

# create necessary folders
mkdir new_dataset
mkdir version

# download pretrained model
# wget \
# $MODEL_METADATA_URI -O version/1/model-metadata.json

wget $MODEL_URI
unzip -q output-model.zip -d version/
rm output-model.zip

# dataset prep
wget $DATASET_URI
unzip -q vh-det-v01.zip -d new_dataset
rm vh-det-v01.zip 