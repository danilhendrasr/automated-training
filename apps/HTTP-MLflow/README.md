# HTTP-MLflow

FastAPI listen http request then run MLflow train.

How to run this project
1. Clone the repo and then open the repo folder.
2. Rename `.env-example` to `.env` then update the file.
2. Build docker image.
```
docker build -t my-http-mlflow-image .
```
3. Run container from that image.
```
docker run \
    -it \
    -p 5005:5005\
    -v /var/run/docker.sock:/var/run/docker.sock \
    --rm \
    --name HTTP-MLflow-TIMC \
    my-http-mlflow-image \
    uvicorn main:app --host 0.0.0.0 --port 5005
```
