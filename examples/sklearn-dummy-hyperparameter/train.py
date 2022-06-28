import random
import time

import mlflow
from sklearn.linear_model import ElasticNet


if __name__ == "__main__":
    mlflow.start_run(run_name="parent")
    param_parent = {}
    metric_parent = {"metric3": 0.0}
    model = None
    list_param_alpha =    [0.3, 0.3, 0.8, 0.4, 0.6, 0.3, 0.6]
    list_param_l1_ratio = [0.3, 0.5, 0.9, 0.7, 0.4, 0.3, 0.4]
    for i in range(7):
        mlflow.start_run(run_name=f"child{i}", nested=True)
        time.sleep(3)
        alpha = random.randint(30, 99) / 100
        l1_ratio = random.randint(30, 99) / 100
        myElasticNet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        metric1 = (alpha + l1_ratio) / 2
        metric2 = (alpha + l1_ratio) * abs(alpha - l1_ratio)
        metric3 = alpha + l1_ratio
        mlflow.log_metric("metric1", metric1)
        mlflow.log_metric("metric2", metric2)
        mlflow.log_metric("metric3", metric3)

        mlflow.end_run()

        # We want the parent is the highest metric3 of child
        if metric3 > metric_parent["metric3"] or i == 0:
            model = myElasticNet
            param_parent["alpha"] = alpha
            param_parent["l1_ratio"] = l1_ratio
            metric_parent["metric1"] = metric1
            metric_parent["metric2"] = metric2
            metric_parent["metric3"] = metric3

    for param_name, param_value in param_parent.items():
        mlflow.log_param(param_name, param_value)
    for metric_name, metric_value in metric_parent.items():
        mlflow.log_metric(metric_name, metric_value)
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()
