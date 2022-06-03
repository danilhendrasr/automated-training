#! This is temporary
import os
import json

def data_prep():
    f = open('data_uri.json')
    uri_obj = json.load(f)
    # os.environ["MODEL_METADATA_URI"] = uri_obj["model_metadata"]
    os.environ["MODEL_URI"] = uri_obj["model"]
    os.environ["DATASET_URI"] = uri_obj["dataset"]
    os.system('chmod +x ./blob-data-extractor.sh')
    os.system('./blob-data-extractor.sh')

    return True