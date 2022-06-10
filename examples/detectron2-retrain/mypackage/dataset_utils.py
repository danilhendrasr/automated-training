import os
import cv2
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

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
    _d = os.listdir(ann_dir)
    # _d.sort()
    for i, f in enumerate(_d):

        if f.split(".")[-1] != "txt":
            continue

        filename = os.path.join(img_dir, ".".join(f.split(".")[:-1]) + ".jpg")
        height, width = cv2.imread(filename).shape[:2]
        ann = load_cvt_dict(os.path.join(ann_dir, f), (height, width))

        record = {
            "file_name": filename,
            "image_id": filename.split("/")[-1],
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
        
        #! dummy limit
        if i == 10:
            break

    return dataset_dicts

def register_dataset(dataset_train, ann_train, dataset_val, ann_val, classes):
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

def register_test_set(dataset, annotation, classes):
     DatasetCatalog.register(
        dataset, lambda: get_dataset_dicts(dataset, annotation)
    )  # registering train class records
     MetadataCatalog.get(dataset).set(
        thing_classes=classes
    ) 
