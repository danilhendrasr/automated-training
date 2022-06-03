# %%
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from mypackage.dataset_utils import register_test_set
import json
import logging
import mlflow
import datetime

from mypackage.test_utils import get_model_from_mlflow, _get_model_from_http_s3, test_set_prep

setup_logger()
cfg = get_cfg()
def preps(ds_uri):
    dataset, ann = test_set_prep(ds_uri)
    with open("classes.json", "r") as cls_file:
        _cls = json.load(cls_file)
    classes = _cls["classes"]
    cfg.MODEL.RETINANET.NUM_CLASSES = len(classes)
    register_test_set(dataset, ann, classes)
    return dataset, ann, classes

def load_model(mlflow_model_uri, from_s3=False):
    if from_s3:
        model_dir = _get_model_from_http_s3(mlflow_model_uri)
    else:
        model_dir = get_model_from_mlflow(mlflow_model_uri)
    cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_dir
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    _predictor = DefaultPredictor(cfg)
    return _predictor
# %%
import random

date_time = str(datetime.datetime.now()).split('.')[0]
model_name = int(random.random()*100)

with mlflow.start_run(run_name=f"model_{model_name}_on_{date_time}"):

    dataset_test, annotation, _ = preps(ds_uri="https://modelrepo-autotraining-poc-ndflx.s3.amazonaws.com/datasets/test_set.zip")
    predictor = load_model("https://modelrepo-autotraining-poc-ndflx.s3.amazonaws.com/models/modelv1/output-model.zip", from_s3=True)
    # %%
    evaluator = COCOEvaluator(dataset_test, output_dir="test_set_eval")
    val_loader = build_detection_test_loader(cfg, dataset_test)  # type: ignore
    eval_result = inference_on_dataset(predictor.model, val_loader, evaluator)
    logging.info("Evaluation results on test set: %s", eval_result)
    mlflow.log_artifacts("test_set_eval", "test_result_modelname")
    print(eval_result)
    
mlflow.end_run()
