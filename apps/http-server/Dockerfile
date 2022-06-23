FROM ubuntu:latest

WORKDIR /app

COPY ./main.py .

COPY .env .

RUN apt-get update

RUN apt-get install -y ca-certificates 

RUN apt-get install -y curl 

RUN apt-get install -y gnupg 

RUN apt-get install -y lsb-release

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update

RUN apt-get install -y docker-ce 

RUN apt-get install -y docker-ce-cli

RUN apt-get install -y containerd.io

RUN apt-get install -y docker-compose-plugin

RUN apt-get install -y python3-pip

RUN pip install mlflow

RUN pip install azure-storage-blob

RUN pip install azure-identity

RUN pip install fastapi

RUN pip install uvicorn

RUN pip install python-dotenv

RUN pip install docker

CMD bash
