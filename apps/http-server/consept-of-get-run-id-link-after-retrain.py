import docker
import dotenv
import mlflow


dotenv.load_dotenv()
local_submitted_run = mlflow.projects.run(
    uri="https://github.com/danilhendrasr/automated-training.git#examples/sklearn-dummy",
    docker_args={"gpus": "device=0"},
    experiment_name="refactor2",
    version="main",
    synchronous=True,
)

mlflow_client = mlflow.tracking.MlflowClient()
run = mlflow_client.get_run(local_submitted_run.run_id)
# Auto remove image that has been used for retrain, so it will not be spamming.
docker_image_name = run.data.tags["mlflow.docker.image.uri"]
docker_client = docker.from_env()
try:
    docker_client.images.remove(docker_image_name)
except docker.errors.ImageNotFound as e:
    print(f"Image '{docker_image_name}' not found")

link = f"http://192.168.103.67:5007/#/experiments/{run.info.experiment_id}/runs/{local_submitted_run.run_id}"
print(link)