# Automated Training

This project is an intern project by Nodeflux's Team C tech intern. It is intended to provide a centralized platform for tracking model training and retraining as well as to automate the retraining process. It is designed to be pluggable wherever and whenever it's needed, therefore we provide an interface in the form of REST API which any other systems can send a POST request to if they wanted to trigger a retraining process.

![image](https://user-images.githubusercontent.com/45989466/178882942-31ee730a-cd5c-44a3-a82c-9fcf703e25de.png)

# Getting Started
There are two components to this project, which is the http server and the mlflow tracking server. Both can be run in a docker container.

## Running the containers

### MLflow Tracking Server App
This is the main tracking server, this needs to be run first.

How to run this app
1. cd to the `apps/mlflow-tracking-server`;
2. Inspect the dockerfile and make changes if your use case requires it. For example, maybe you're using google cloud storage as the artifact storage, in which case you can refer to [the docs](https://mlflow.org/docs/latest/tracking.html#google-cloud-storage) to know what changes needs to be made;
3. Build the docker image:
```bash
docker build -t mlflow-tracking-server .
```
4. Run the container.
```bash
docker run \
    -itd \
    -p 5007:5000\
    --rm \
    --name mlflow-tracking-server \
    mlflow-tracking-server
```

### HTTP Server App
This HTTP server is built using FastAPI and its role is to listen to http requests and then trigger the retraining process.

How to run this app
1. Clone the repo and then open this folder.
2. Create `.env` file or rename `.env.example` to `.env` then update the file. 
`MLFLOW_TRACKING_URI` points to MLflow tracking server (:5007), `AZURE_STORAGE_ACCESS_KEY` untuk blob storage, dan `webhook_url` adalah URL [webhook discord](https://discord.com/developers/docs/resources/webhook#get-channel-webhooks)
```
MLFLOW_TRACKING_URI=""
AZURE_STORAGE_ACCESS_KEY=""
webhook_url=""
```
3. Build docker image.
```
docker build -t http-mlflow-training-image .
```
4. Run container from that image.
```
docker run \
    -it \
    -p 5005:5005\
    -v /var/run/docker.sock:/var/run/docker.sock \
    --rm \
    --name HTTP-MLflow-trainig \
    http-mlflow-training-image \
    uvicorn main:app --host 0.0.0.0 --port 5005
```
5. `MLflow Training server` container will run at port `5005`

## Preparing Your Project

### Prerequisite
0. Build system
1. Fully working `train.py`, as a retraining script
2. Environment (docker image di Docker Hub or local) + azure library
3. Disimpan dalam Github repository atau menggunakan [template repo](https://github.com/hamzahmhmmd/MLflow-retrain/generate)


### Create MLproject file
The first thing you need to do is to create an MLproject file, you can check out the [examples directory](https://github.com/danilhendrasr/automated-training/blob/main/examples/detectron2-retrain/MLproject) or [the docs](https://mlflow.org/docs/latest/projects.html#mlproject-file) for references. Then follow the following steps when done.

1. Define docker image as the environment, like the following:
```
docker_env:
  image: training-container:detectron2 # just example
```

2. Define the retraining script as main entrypoint
```
entry_points:
  main: 
    command: "python train.py"
```

3. Define the required parameters, every parameter should be passed to the training script as an argument.
```
entry_points:
  main: 
    parameters:
      model_name : {type: string, default: "default"}
      experiment_name : {type: string, default: "default"}
      lower_is_better : {type: bool, default: False}
      primary_metric : {type: string, default: "mAP"}
      base_model : {type: string, default: "None"}
      dataset_repo  : {type: string, default: "not_used"}
    command: 
      "python train.py |
            --model_name {model_name} \
            --experiment_name {experiment_name} \
            --lower {lower_is_better} \
            --p_metric {primary_metric} \
            --base_model {base_model} \
            --dataset_repo {dataset_repo}"
```
- `model_name` is name of the model already registered in the MLflow model registry
- `experiment_name` is the name of the project, to distinguish it from other projects
- `lower_is_better` is used by the system when doing metric comparison (comparing `primary_metric`) to determine whether lower value is better or the opposite
- `base_model` should be filled with the base model's *MLflow run ID*
- `dataset_repo` shoudl be filled with a Git repo URL which contains the dataset, we are expecting the dataset versioning to be done using DVC, refer to [this repo](https://github.com/millenia911/vehicle-detection-timc.git) for reference

### A Little Changes to `train.py`
1. Import utility functions
We provide utility functions, for now the only utility function is a function to automatically register a new version of a model if the primary metric is better than the previous version. Just copy and paste the `mypackage` directory to your project, the result structure should look like this:
```
ROOT
|- train.py
|- MLproject
|- mypackage
   |- utility.py
```

2. Import MLflow
```python
import mlflow

#! your training script
```

3. Catch parameters as arguments
```python
from mypackage.prep_utils import data_and_model_prep
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", required=True,)
parser.add_argument("--lower", required=False,)
parser.add_argument("--p_metric", required=True,)
parser.add_argument("--base_model", required=True,)
parser.add_argument("--dataset_repo", required=True,)
args , _ = parser.parse_known_args()

base_model = args.base_model
dataset_repo = args.dataset_repo
model_name = args.model_name # String, model name
p_metric = args.p_metric # String, primary metric name
lower = args.lower # String, true of false
if lower is not None and lower.lower() == "false":
    lower = False
else:
    lower = True

#! your training script
```

4. Download the base model dan datasets
```python
if base_model != "None":
    mlflow.artifacts.download_artifacts(run_id='base_model')
else:
    # your methods to download the base model

if dataset_repo != "not_used":
    data_and_model_prep(repo_url=dataset_repo, ds_origin="dvc_repo")
else:
    # your methods to download datasets

#! your training script
```

5. Log model
```python
from mypackage import utility

#! your training script

client = mlflow.tracking.MlflowClient()
auto_compare_and_register(
  model=model_object,
  eval_metric=primary_metric_object,
  model_name=model_name, # String, model name
  lower=lower, # True or False
  p_metric=p_metric, # String, primary metric name
  client=client,
  mlflow_model=mlflow.models, # ref: https://mlflow.org/docs/latest/python_api/mlflow.models.html
)
```

6. Log parameters and metrics to tracking server
```python
#! your training script

mlflow.log_param("param_name", "param_value")
#! add more params

mlflow.log_metric(p_metric,       metric1_object) # log primary object
mlflow.log_metric("metric_name2", metric1_object)
mlflow.log_metric("metric_name3", metric1_object)
#! add more metrics

mlflow.log_artifact("train.py")
#! add more files
```

### Automate and Log Hyperparameter Search (Optional)
Untuk mengadopsi fitur ini ada beberapa perbedaan pada file `train.py`, tapi sebelum itu pastikan telah melakukan langkah-langkah diatas sebelumnya.
Perubahan tersebut berada sebelum training loop anda, seperti kode dibawah.
Contoh implementasi ada pada file [example ini](https://github.com/danilhendrasr/automated-training/blob/main/examples/sklearn-dummy-hyperparameter/train.py).

```
from mypackage import utility
    
mlflow.start_run(run_name="parent")
param_parent = {}
metric_parent = {"metric3": 0.0}
    
list_param_alpha = [0.43, 0.82, 0.46]
list_param_l1_ratio = [0.79, 0.73, 0.49]
list_param = get_combination(list_param_alpha, list_param_l1_ratio)
    
for i, param in enumerate(list_param):
    mlflow.start_run(run_name=f"child{i}", nested=True)
        
    alpha = param[0]
    l1_ratio = param[1]
        
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
        
    #! your training script 
```

## Hit API to Trigger Retraining
ini adalah contoh POST request ke mlflow retraining server (misal :5005)
1. Get GitHub repo URI, untuk public repo `https://git@github.com/user/repo_name.git` dan untuk private repo `https://{your_personal_access_token}@github.com/user/repo_name.git`
2. Send POST request ke `http://ip:5005/retrain` dengan body seperti contoh berikut 
```
{
  "experiment_name": "name_of_experiment",
  "git_uri": "https://git@github.com/user/repo_name.git",
  "MLproject_location": "path/to/MLproject/directoy",
  "commit_hash": "main",
  "dataset_uri": "", #optional
  "base_model" : ""  #optional # mlflow run id
}
```
3. Anda juga dapat menggunakan UI Swager untuk melakukan request secara manual, engan mengunjungi `http://ip:5005/docs`
