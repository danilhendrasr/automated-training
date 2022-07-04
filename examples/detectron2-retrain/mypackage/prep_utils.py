import os
import json
import shutil

def data_and_model_prep(dst_dir:str="new_dataset", repo_url:str="", ds_origin:str="http", json_dir:str="data_uri.json"):
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs("version", exist_ok=True)
    download_model(json_dir=json_dir)

    if ds_origin == "http":
        from_http_in_json(dst_dir, json_dir)

    elif ds_origin == "dvc_repo":
        from_dvc_repo(repo_url, dst_dir)

    else: raise ValueError("'http' and 'dvc_repo' are the only supported options")
    return

def from_dvc_repo(repo_url:str, dst_dir:str="./"):
    if repo_url.split(".")[-1] != "git":
        raise ValueError("repo is not a git uri")

    repo_name = dvc_pull(repo_url)
    repo_files = os.listdir(repo_name)

    pulled_dataset = [x for x in repo_files if x.endswith(".dvc") and not x.startswith(".")]
    default=0
    pulled_dataset = pulled_dataset[default]
    pulled_dataset_path = pulled_dataset.split(".")[0]

    if os.path.isdir(os.path.join(repo_name, pulled_dataset_path)):
        print(os.listdir(os.path.join(repo_name, pulled_dataset_path)))
        for ds in os.listdir(os.path.join(repo_name, pulled_dataset_path)):
            shutil.copytree(src=os.path.join(repo_name, pulled_dataset_path, ds),
                            dst=os.path.join(dst_dir, ds))
    else:
        shutil.copy(src=os.path.join(repo_name, pulled_dataset_path),
                    dst=os.path.join(dst_dir))

    shutil.rmtree(repo_name)
    return

def dvc_pull(repo_url):
    repo_name = repo_url.split("/")[-1].split(".")[0]
    cwd = os.getcwd()
    os.system(f"git clone {repo_url}")
    os.chdir(repo_name)
    os.system("dvc pull")
    os.chdir(cwd)
    return repo_name

def from_http_in_json(dst_dir, json_dir:str="data_uri.json"):
    f = open(json_dir)
    uri_obj = json.load(f)
    os.environ["MODEL_URI"] = uri_obj["model"]
    os.environ["DATASET_URI"] = uri_obj["dataset"]
    os.environ["DST_DIR"] = dst_dir
    os.system('chmod +x ./dataset-extractor.sh')
    os.system('./dataset-extractor.sh')
    f.close()
    return 0

def download_model(uri=None, json_dir:str="data_uri.json"):
    if uri is None:
        f = open(json_dir)
        uri_obj = json.load(f)
        os.environ["MODEL_URI"] = uri_obj["model"]
    else: os.environ["MODEL_URI"] = uri
    os.system('chmod +x ./model-downloader.sh')
    os.system('./model-downloader.sh')