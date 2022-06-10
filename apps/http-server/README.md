# HTTP-MLflow

FastAPI listen http request then run MLflow train.

How to run this project
1. Clone the repo and then open the repo folder.
2. Rename `.env.example` to `.env` then update the file.
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
4. Navigate to `http://server.name/docs/` to send request using swager UI
5. or Send `POST` request (using our example)
```
curl -X 'POST' \
  'http://server.name/retrain' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "experiment_name": "name-of-your-experiment",
  "get_uri": "https://git@github.com/danilhendrasr/automated-training.git#examples/detectron2-retrain",
  "get_commit_id": "main"
}'
```
