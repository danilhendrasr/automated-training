from detectron2.engine import HookBase
import mlflow
import torch
import os

class MLflowHook(HookBase):
    """
    A custom hook class that logs artifacts, metrics, and parameters to MLflow.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

    def before_train(self):
        with torch.no_grad():
            mlflow.start_run()
            mlflow.set_tag("mlflow.note.content",
                           self.cfg.MLFLOW.RUN_DESCRIPTION)
            for k, v in self.cfg.items():
                try:
                    if isinstance(v, str):
                        if len(v) > 10:
                            continue
                    mlflow.log_param(k, v)
                except:
                    print(f"FAILED TO LOG PARAM => {k}")

    def after_step(self):
        with torch.no_grad():
            latest_metrics = self.trainer.storage.latest()
            for k, v in latest_metrics.items():
                mlflow.log_metric(key=k, value=v[0], step=v[1])

    def after_train(self):
        with torch.no_grad():
            with open(os.path.join(self.cfg.OUTPUT_DIR, "model-config.yaml"), "w") as f:
                f.write(self.cfg.dump())
            mlflow.log_artifacts(self.cfg.OUTPUT_DIR)