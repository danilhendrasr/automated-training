import os

import dotenv
import mlflow
import requests
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel


dotenv.load_dotenv()
app = FastAPI()
webhook_url = os.environ["webhook_url"]


class Parameters(BaseModel):
    experiment_name: str
    get_uri: str
    get_commit_id: str = "main"


@app.get("/")
async def root():
    return {"message": "Hello, please go to /docs."}


@app.post("/retrain")
async def retrain(parameters: Parameters):
    requests.post(webhook_url, json={"content": "MLFlow retrain running"})
    try:
        mlflow.projects.run(
            uri=parameters.get_uri,
            docker_args={"gpus": "device=0"},
            experiment_name=parameters.experiment_name,
            version=parameters.get_commit_id,
        )
        requests.post(webhook_url, json={"content": "MLFlow retrain succeeded"})
    except Exception as e:
        requests.post(
            webhook_url, json={"content": f"MLFlow retrain failed : {str(e)}"}
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"message": "Success"}
