import mlflow

def auto_compare_and_register(model, eval_metric, model_name, lower, p_metric, client, mlflow_model):
    """Do auto compare eval_metric with latest metric on the latest version model registered"""
    try:
        latest_version_run_id = client.get_latest_versions(model_name)[0].run_id
    except mlflow.exceptions.RestException:
        mlflow_model.log_model(model, 'model', registered_model_name=model_name)
        return
    try:
        latest_p_metric = client.get_metric_history(latest_version_run_id, p_metric)[0].value
    except IndexError:
        raise ValueError("wrong metric name")
    # Dibawah ini bisa di refactor, sementara ku tulis begini biar jelas.
    # clear is better than clever
    if lower is True:
        if eval_metric > latest_p_metric:
            mlflow_model.log_model(model, 'model')
        else:
            mlflow_model.log_model(model, 'model', registered_model_name=model_name)
    elif lower is False:
        if eval_metric > latest_p_metric:
            mlflow_model.log_model(model, 'model', registered_model_name=model_name)
        else:
            mlflow_model.log_model(model, 'model')