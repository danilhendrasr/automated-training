# Automated Training

This project is an intern project by Nodeflux's Team C tech intern. It is intended to provide a centralized platform for tracking model training and retraining as well as to automate the retraining process. It is designed to be pluggable wherever and whenever it's needed, therefore we provide an interface in the form of REST API which any other systems can send a POST request to if they wanted to trigger a retraining process.

![image](https://user-images.githubusercontent.com/45989466/178882942-31ee730a-cd5c-44a3-a82c-9fcf703e25de.png)

# Getting Started
There are two components to this project, which is the http server and the mlflow tracking server. Both can be run in a docker container.

## MLflow Tracking Server
This is the main tracking server, this needs to be run first.

1. cd to the `apps/mlflow-tracking-server`;
2. Inspect the dockerfile and make changes if your use case requires it. For example, maybe you're using google cloud storage as the artifact storage, in which case you can refer to [the docs](https://mlflow.org/docs/latest/tracking.html#google-cloud-storage) to know what changes needs to be made;
3. Build the docker image:
```bash
docker build -t mlflow-tracking-server .
```
4. Run the container.
```bash
docker run \
    -itd \
    -p 5007:5000\
    --rm \
    --name mlflow-tracking-server \
    mlflow-tracking-server
```

## HTTP Server

This HTTP server is built using FastAPI and its role is to listen to http requests and then trigger the retraining process.

How to run this project
1. Clone the repo and then open the directory.
2. Rename `.env.example` to `.env` then update the file.
2. Build docker image.
```
docker build -t mlflow-training-server .
```
3. Run container from that image.
```
docker run \
    -it \
    -p 5005:5005\
    -v /var/run/docker.sock:/var/run/docker.sock \
    --rm \
    --name mlflow-training-server \
    mlflow-training-server \
    uvicorn main:app --host 0.0.0.0 --port 5005
```
