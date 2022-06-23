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
    git_uri: str
    folder_location: str = ""
    commit_hash: str = "main"


@app.get("/")
async def root():
    return {"message": "Hello, please go to /docs."}


@app.post("/retrain")
async def retrain(parameters: Parameters):
    content = "MLflow retrain running\n```\n{\n"
    for k, v in parameters:
        content += f'\t"{k}": "{v}",\n'
    content += "}\n```"
    requests.post(webhook_url, json={"content": content})
    uri = parameters.git_uri
    if parameters.folder_location != "":
        uri += f"#{parameters.folder_location}"
    try:
        mlflow.projects.run(
            uri=uri,
            docker_args={"gpus": "device=0"},
            experiment_name=parameters.experiment_name,
            version=parameters.commit_hash,
        )
        requests.post(webhook_url, json={"content": "MLflow retrain succeeded"})
    except Exception as e:
        requests.post(
            webhook_url, json={"content": f"MLflow retrain failed : {str(e)}"}
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"message": "Success"}
