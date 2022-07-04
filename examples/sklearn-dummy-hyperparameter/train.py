import time

import mlflow
from sklearn.linear_model import ElasticNet


def auto_compare_and_register(
    model, eval_metric, model_name, lower, p_metric, client, mlflow_model
):
    """Do auto compare eval_metric with latest metric on the latest version model registered"""
    try:
        latest_version_run_id = client.get_latest_versions(model_name)[0].run_id
    except mlflow.exceptions.RestException:
        mlflow_model.log_model(model, "model", registered_model_name=model_name)
        return
    try:
        latest_p_metric = client.get_metric_history(latest_version_run_id, p_metric)[0].value
    except IndexError:
        raise ValueError("wrong metric name")
    # Dibawah ini bisa di refactor, sementara ku tulis begini biar jelas.
    # clear is better than clever
    if lower is True:
        if eval_metric > latest_p_metric:
            mlflow_model.log_model(model, "model")
        else:
            mlflow_model.log_model(model, "model", registered_model_name=model_name)
    elif lower is False:
        if eval_metric > latest_p_metric:
            mlflow_model.log_model(model, "model", registered_model_name=model_name)
        else:
            mlflow_model.log_model(model, "model")


def get_combination(
    list_param_alpha: list[float], list_param_l1_ratio: list[float]
) -> list[tuple[float, float]]:
    list_param = []
    for alpha in list_param_alpha:
        for l1_ratio in list_param_l1_ratio:
            p = (alpha, l1_ratio)
            list_param.append(p)
    return list_param


if __name__ == "__main__":
    mlflow.start_run(run_name="parent")
    param_parent = {}
    metric_parent = {"metric3": 0.0}
    model = None
    list_param_alpha = [0.43, 0.82, 0.46]
    list_param_l1_ratio = [0.79, 0.73, 0.49]
    list_param = get_combination(list_param_alpha, list_param_l1_ratio)
    for i, param in enumerate(list_param):
        mlflow.start_run(run_name=f"child{i}", nested=True)
        time.sleep(1)
        # region params
        alpha = param[0]
        l1_ratio = param[1]
        myElasticNet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        # endregion params

        # region train & evaluate
        # Train & evaluate code here.
        #
        # For this dummy example, let's say after evaluate we got metric.
        metric1 = (alpha + l1_ratio) / 2
        metric2 = (alpha + l1_ratio) * abs(alpha - l1_ratio)
        metric3 = alpha + l1_ratio
        mlflow.log_metric("metric1", metric1)
        mlflow.log_metric("metric2", metric2)
        mlflow.log_metric("metric3", metric3)
        # endregion train & evaluate

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

    client = mlflow.tracking.MlflowClient()
    auto_compare_and_register(
        model=model,
        eval_metric=metric_parent["metric3"],
        model_name="SklearnElasticnetHyperparameterDummy",
        lower=False,
        p_metric="metric3",
        client=client,
        mlflow_model=mlflow.sklearn,
    )
    mlflow.end_run()
