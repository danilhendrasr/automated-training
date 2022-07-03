#!/bin/bash

# dataset prep
wget $DATASET_URI
unzip -q vh-det-v01.zip -d $DST_DIR
rm vh-det-v01.zip 