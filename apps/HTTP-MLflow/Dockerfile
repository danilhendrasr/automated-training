FROM ubuntu:latest

WORKDIR /app

COPY ./main.py .

COPY .env .

RUN apt-get update

RUN apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update && apt-get install -y docker-ce docker-ce-cli \
    containerd.io docker-compose-plugin

RUN apt-get update && apt-get install python3-pip -y

RUN pip install mlflow azure-storage-blob azure-identity fastapi uvicorn python-dotenv

CMD bash