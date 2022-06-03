import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class TrainerWithCocoEVal(DefaultTrainer):
    """
    A custom trainer class that evaluates the model on the validation set every `_C.TEST.EVAL_PERIOD` iterations.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(cfg.OUTPUT_DIR_VALIDATION_SET_EVALUATION,
                        exist_ok=True)

        return COCOEvaluator(dataset_name, distributed=False, output_dir=cfg.OUTPUT_DIR_VALIDATION_SET_EVALUATION)
