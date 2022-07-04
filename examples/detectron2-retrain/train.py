import datetime
import json
import os
import logging
import random
import argparse
import mlflow
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from mypackage import utility
from mypackage.dataset_utils import *
from mlflow_hooks import *
from trainer import *
from mypackage.prep_utils import data_and_model_prep

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Nama model untuk diregistrasi.")
    parser.add_argument("--experiment_name", required=True, help="Digunakan untuk separate dengan project / repo yang lain.")
    parser.add_argument("--lower", required=False, help="Default True. Diguankan untuk perbandingan metric yang diinginkan\n Jika True, maka akan dilakukan registrasi model bila metric lebih rendah daripada metric di registered model")
    parser.add_argument("--p_metric", required=True, help="Nama metric yang akan dibandingkan.")
    parser.add_argument("--dataset_repo", required=False, default="not_used", help="A DVC repo to pull datased. If DVC is not used, set to `not_used`. Default is `not_used`")
    parser.add_argument("--base_model", required=True, help="base model origin")
    return parser.parse_args()

def do_autoregister(args, predictor, eval_result):
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

if __name__ == "__main__":
    setup_logger()
    args_ = arg_parser()

    if args_.dataset_repo != "not_used" or args_.dataset_repo is not None:
        data_and_model_prep(repo_url=args_.dataset_repo, ds_origin="dvc_repo")
    else: data_and_model_prep()
    

    # get config, set config for MLFlow
    cfg = get_cfg()
    cfg.MLFLOW = CfgNode()
    cfg.MLFLOW.RUN_DESCRIPTION = "None"

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

    register_dataset(dataset_train, 
                    ann_train, 
                    dataset_val, 
                    ann_val, 
                    classes)

    # setup training parameters
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

    # setup traing config
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
    cfg.SOLVER.STEPS = [] 
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

    # setup and begin training
    mlflow_hook = MLflowHook(cfg)
    trainer = TrainerWithCocoEVal(cfg)
    trainer.register_hooks(hooks=[mlflow_hook])
    trainer.resume_or_load(resume=False)
    trainer.train() 

    # setup test
    setup_logger(output=os.path.join(cfg.OUTPUT_DIR_TEST_SET_EVALUATION, "evaluation-log.txt"))
    cfg.MODEL.WEIGHTS = model_dir
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(dataset_val, output_dir=cfg.OUTPUT_DIR_TEST_SET_EVALUATION)
    val_loader = build_detection_test_loader(cfg, dataset_val) 
    eval_result = inference_on_dataset(predictor.model, val_loader, evaluator)
    logging.info("Evaluation results on test set: %s", eval_result)

    for k, v in eval_result["bbox"].items():
        mlflow.log_metric(f"Test Set {k}", v, step=0)

    # Generate test results sample
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

    # log test result
    mlflow.log_artifacts(cfg.OUTPUT_DIR_TEST_SET_EVALUATION, "test-set-evaluation")
    mlflow.log_text(str(eval_result), "test-set-evaluation/coco-metrics.txt")
    
    do_autoregister(args_, predictor, eval_result)

    # write metadata
    # metadata["time_stamp"] = str(datetime.datetime.now())
    # metadata["version"] = "trial"
    # metadata["eval"]["result"]["bbox"] = eval_result["bbox"]
    # with open(model_metadata_dir, "w") as outfile:
    #     json.dump(metadata, outfile, indent=4)

    mlflow.end_run()
