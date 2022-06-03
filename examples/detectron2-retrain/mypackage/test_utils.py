import mlflow
from mlflow.artifacts import download_artifacts
import os



def get_model_from_mlflow(uri):
    path = download_artifacts(artifact_uri=uri, 
                              dst_path="./")
    return path

def _get_model_from_http_s3(uri):
    os.system(f"wget {uri}")
    os.system("unzip -q output-model.zip")
    return "./output/model_final.pth"

def test_set_prep(ds_uri):
    _prefix = "test_set"
    os.makedirs(_prefix, exist_ok=True)
    os.system(f"wget {ds_uri} -O {_prefix}.zip")
    os.system(f"unzip -q {_prefix}.zip -d {_prefix}")
    ls = os.listdir(_prefix)
    for _f in ls:
        if _f.startswith("data"):
            _ds = _prefix+"/"+_f
        elif _f.startswith("ann"):
            _ann = _prefix+"/"+_f
    return _ds, _ann