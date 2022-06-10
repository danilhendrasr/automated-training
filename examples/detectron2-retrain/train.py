# %%
import datetime
import json
import os
import shutil
import logging
import random
from pathlib import Path
import argparse

import mlflow
# os.environ['AZURE_STORAGE_ACCESS_KEY']= 
# mlflow.set_tracking_uri('http://192.168.103.67:5001/')
# mlflow.set_experiment("training")


from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer


from mypackage import utility
from mypackage.dataset_utils import *
from mlflow_hooks import *
from trainer import *

from mypackage.temporary_prep import data_prep
data_prep()

# %%
setup_logger()

# get config, set config for MLFlow
cfg = get_cfg()
cfg.MLFLOW = CfgNode()
cfg.MLFLOW.RUN_NAME = str(int(random.random()*100000)) 
cfg.MLFLOW.RUN_DESCRIPTION = "ini ceritanya deskripsi"
##


# %%
new_dataset_dir = "new_dataset"
dataset_train = os.path.join(new_dataset_dir, "dataset_train")
dataset_val = os.path.join(new_dataset_dir, "dataset_val")
ann_train = os.path.join(new_dataset_dir, "ann_train")
ann_val = os.path.join(new_dataset_dir, "ann_val")

new_version_dir = utility.create_new_version_dir()
model_metadata_dir = os.path.join(new_version_dir, "model-metadata.json")
with open(model_metadata_dir) as mtdata:
    metadata = json.load(mtdata)
classes = metadata["classes"]

# %%
register_dataset(dataset_train, 
                 ann_train, 
                 dataset_val, 
                 ann_val, 
                 classes)

# %%
batch_size = 4
base_rl = 0.001
iteration = 10
warmup_iters = 1
freeze_at = 5

metadata["train_cfg"]["batch_size"] = batch_size 
metadata["train_cfg"]["base_rl"] = base_rl 
metadata["train_cfg"]["iter"] = iteration 
metadata["train_cfg"]["warmup_iters"] = warmup_iters 
metadata["train_cfg"]["freeze_at"] = freeze_at

# %%
output_dir = os.path.join(new_version_dir, "output")
cfg.OUTPUT_DIR = output_dir

cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
)

cfg.DATASETS.TRAIN = (dataset_train,)
cfg.DATASETS.TEST = (dataset_val,)
cfg.DATALOADER.NUM_WORKERS = 2
model_dir = os.path.join(output_dir, "model_final.pth")
cfg.MODEL.WEIGHTS = model_dir
cfg.SOLVER.IMS_PER_BATCH = batch_size
cfg.SOLVER.BASE_LR = base_rl
cfg.SOLVER.WARMUP_ITERS = warmup_iters
cfg.SOLVER.MAX_ITER = iteration
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.RETINANET.NUM_CLASSES = len(classes)
cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at
cfg.OUTPUT_DIR_VALIDATION_SET_EVALUATION = os.path.join(
        cfg.OUTPUT_DIR, "validation-set-evaluation")
cfg.OUTPUT_DIR_TEST_SET_EVALUATION = os.path.join(
        cfg.OUTPUT_DIR, "test-set-evaluation")
cfg.TEST.EVAL_PERIOD = 3
os.makedirs(cfg.OUTPUT_DIR_VALIDATION_SET_EVALUATION, exist_ok=True)
os.makedirs(cfg.OUTPUT_DIR_TEST_SET_EVALUATION, exist_ok=True)

setup_logger(output=os.path.join(cfg.OUTPUT_DIR, "training-log.txt"))

# %%
mlflow_hook = MLflowHook(cfg)

trainer = TrainerWithCocoEVal(cfg)
trainer.register_hooks(hooks=[mlflow_hook])
trainer.resume_or_load(resume=False)
trainer.train()

# %%
setup_logger(output=os.path.join(cfg.OUTPUT_DIR_TEST_SET_EVALUATION, "evaluation-log.txt"))
cfg.MODEL.WEIGHTS = model_dir
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator(dataset_val, output_dir=cfg.OUTPUT_DIR_TEST_SET_EVALUATION)
val_loader = build_detection_test_loader(cfg, dataset_val)  # type: ignore
eval_result = inference_on_dataset(predictor.model, val_loader, evaluator)
logging.info("Evaluation results on test set: %s", eval_result)

for k, v in eval_result["bbox"].items():
    mlflow.log_metric(f"Test Set {k}", v, step=0)

# %% Generate test results sample
# ! Just temporary
# TODO: split new test set
dataset_metadata = MetadataCatalog.get(dataset_train)
dataset_dicts = get_dataset_dicts(dataset_val, ann_val)
sample_path = os.path.join(cfg.OUTPUT_DIR_TEST_SET_EVALUATION,"test_sample")
os.makedirs(sample_path,exist_ok=True)
i=0
for d in random.sample(dataset_dicts, 8):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im) 
    v = Visualizer(im[:, :, ::-1],
                   metadata=dataset_metadata, 
                   scale=2, 
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(f"{sample_path}/sample{i}.jpg", out.get_image()[:, :, ::-1])
    i += 1
# %%

mlflow.log_artifacts(cfg.OUTPUT_DIR_TEST_SET_EVALUATION, "test-set-evaluation")
mlflow.log_text(str(eval_result), "test-set-evaluation/coco-metrics.txt")
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True, help="Nama model untuk diregistrasi.")
parser.add_argument("--experiment_name", required=True, help="Digunakan untuk separate dengan project / repo yang lain.")
parser.add_argument("--lower", required=False, help="Default True. Diguankan untuk perbandingan metric yang diinginkan\n Jika True, maka akan dilakukan registrasi model bila metric lebih rendah daripada metric di registered model")
parser.add_argument("--p_metric", required=True, help="Nama metric yang akan dibandingkan.")
args = parser.parse_args()

model_name = args.model_name
# Ini exp_name tidak terpakai, tetap kutulis karena ada di params MLproject
exp_name = args.experiment_name
lower = args.lower
if lower is not None and lower.lower() == "false":
    lower = False
else:
    lower = True
p_metric = args.p_metric
client = mlflow.tracking.MlflowClient()
utility.auto_compare_and_register(
    model = predictor.model,
    eval_metric=eval_result["bbox"][p_metric],
    model_name=model_name,
    lower=lower,
    p_metric=f"Test Set {p_metric}",
    client=client,
    mlflow_model=mlflow.pytorch
)
# %%
metadata["time_stamp"] = str(datetime.datetime.now())
version = new_version_dir.split("/")[-1]
metadata["version"] = version
metadata["eval"]["result"]["bbox"] = eval_result["bbox"]
with open(model_metadata_dir, "w") as outfile:
    json.dump(metadata, outfile, indent=4)

# write all versions benchmark
# utility.write_all_versions_benchmark()
# endregion Eval


# %%
mlflow.end_run()
