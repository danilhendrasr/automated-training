import datetime
import json
import os
import shutil
from pathlib import Path

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

from mypackage import utility


setup_logger()


# input
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


# region Dataset prep
def load_cvt_dict(file_annot, imsize):
    (imh, imw) = imsize
    obj_list = []
    _f = open(file_annot)
    _f = _f.read().split("\n")
    for labels in _f:
        if labels != "":
            split_annot = labels.split(" ")
            obj = {"class": split_annot[0]}
            for i, key in enumerate(["x", "y", "w", "h"]):
                if key in ["x", "w"]:
                    obj[key] = int(float(split_annot[i + 1]) * imw)  # type: ignore
                if key in ["y", "h"]:
                    obj[key] = int(float(split_annot[i + 1]) * imh)  # type: ignore

            obj_list.append(obj)

    return obj_list


def get_dataset_dicts(img_dir, ann_dir):
    dataset_dicts = []
    for i, f in enumerate(os.listdir(ann_dir)):

        if f.split(".")[-1] != "txt":
            continue

        filename = os.path.join(img_dir, ".".join(f.split(".")[:-1]) + ".jpg")
        height, width = cv2.imread(filename).shape[:2]
        ann = load_cvt_dict(os.path.join(ann_dir, f), (height, width))

        record = {
            "file_name": filename,
            "image_id": i,
            "height": height,
            "width": width,
        }

        bbox = ann
        objs = []
        for bx in bbox:
            obj = {
                "bbox": [
                    int(bx["x"] - (0.5 * bx["w"])),
                    int(bx["y"] - (0.5 * bx["h"])),
                    bx["w"],
                    bx["h"],
                ],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": int(bx["class"]),
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


DatasetCatalog.register(
    dataset_train, lambda: get_dataset_dicts(dataset_train, ann_train)
)  # registering train class records
MetadataCatalog.get(dataset_train).set(
    thing_classes=classes
)  # set class to datasetcatalog

DatasetCatalog.register(
    dataset_val, lambda: get_dataset_dicts(dataset_val, ann_val)
)  # registering val class records
MetadataCatalog.get(dataset_val).set(
    thing_classes=classes
)  # set class to datasetcatalog

batch_size = metadata["train_cfg"]["batch_size"]
base_rl = metadata["train_cfg"]["base_rl"]
iteration = metadata["train_cfg"]["iter"]
warmup_iters = metadata["train_cfg"]["warmup_iters"]
# endregion Dataset prep


# region Training
cfg = get_cfg()
output_dir = os.path.join(new_version_dir, "output")
cfg.OUTPUT_DIR = output_dir
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = (dataset_train,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
model_dir = os.path.join(output_dir, "model_final.pth")
cfg.MODEL.WEIGHTS = model_dir
cfg.SOLVER.IMS_PER_BATCH = batch_size
cfg.SOLVER.BASE_LR = base_rl
cfg.SOLVER.WARMUP_ITERS = warmup_iters
cfg.SOLVER.MAX_ITER = iteration
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.RETINANET.NUM_CLASSES = len(classes)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
# endregion Training


# region Eval
cfg.MODEL.WEIGHTS = model_dir
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator(dataset_val, output_dir=output_dir)
val_loader = build_detection_test_loader(cfg, dataset_val)  # type: ignore
eval_result = inference_on_dataset(predictor.model, val_loader, evaluator)

# write version benchmark
metadata["time_stamp"] = str(datetime.datetime.now())
version = new_version_dir.split("/")[-1]
metadata["version"] = version
metadata["eval"]["result"]["bbox"] = eval_result["bbox"]
with open(model_metadata_dir, "w") as outfile:
    json.dump(metadata, outfile)

# write all versions benchmark
utility.write_all_versions_benchmark()
# endregion Eval


# region backup new dataset
os.rename(new_dataset_dir, version)
shutil.move(version, os.path.join("all_dataset", "version"))
Path(dataset_train).mkdir(parents=True)
Path(dataset_val).mkdir(parents=True)
Path(ann_train).mkdir(parents=True)
Path(ann_val).mkdir(parents=True)
# endregion

print()
