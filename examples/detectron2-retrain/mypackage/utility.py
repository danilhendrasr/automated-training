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
