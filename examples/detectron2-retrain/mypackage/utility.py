import os
import shutil


def create_new_version_dir():
    """copy latest version model dir to new version model dir"""
    version_dir = "version"
    version_list = [int(i) for i in os.listdir(version_dir)]
    latest_version = max(version_list)
    latest_version_dir = os.path.join(version_dir, str(latest_version))
    new_version_dir = os.path.join(version_dir, str(latest_version + 1))
    print(f"copy {latest_version_dir} to {new_version_dir}")
    shutil.copytree(latest_version_dir, new_version_dir)
    return new_version_dir


def auto_compare_and_register(model, eval_metric, model_name, lower, p_metric, client, mlflow_model):
    """Do auto compare eval_metric with latest metric on the latest version model registered"""
    latest_version_run_id = client.get_latest_versions(model_name)[0].run_id
    latest_p_metric = client.get_metric_history(latest_version_run_id, p_metric)[0].value
    # Dibawah ini bisa di refactor, sementara ku tulis begini biar jelas.
    # clear is better than clever
    if lower is True:
        if eval_metric > latest_p_metric:
            mlflow_model.log_model(model, 'model')
        else:
            mlflow_model.log_model(model, 'model', registered_model_name=model_name)
    elif lower is False:
        if eval_metric > latest_p_metric:
            mlflow_model.log_model(model, 'model', registered_model_name=model_name)
        else:
            mlflow_model.log_model(model, 'model')

# ! to be removed
# def write_all_versions_benchmark():
#     """iterate through all version then grep the eval result bbox"""
#     benchmark = {}
#     list_version_dir = os.listdir("version")
#     for i in list_version_dir:
#         model_metadata_path = os.path.join("version", i, "model-metadata.json")
#         with open(model_metadata_path) as mtdata:
#             metadata = json.load(mtdata)
#         benchmark[i] = metadata["eval"]["result"]["bbox"]

#     with open("benchmark.json", "w") as f:
#         json.dump(benchmark, f)
