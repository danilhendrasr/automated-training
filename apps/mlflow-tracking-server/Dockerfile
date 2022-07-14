FROM continuumio/miniconda3:latest

WORKDIR /mlflow/

# TODO: Install dependencies which you need.
# For example, install azure-storage-blob if you need to store the artifacts
# in Azure Blob Storage because MLflow doesn't depend on it by default.
#
# If you use other providers, check out the docs and install the required packages
# accordingly. 
RUN pip install mlflow>=1.0 \
  && pip install psycopg2-binary \
  && pip install azure-storage-blob==12.3.0 \
  && pip install numpy==1.21.2 \
  && pip install scipy \
  && pip install pandas==1.3.3 \
  && pip install scikit-learn==0.24.2 \
  && pip install cloudpickle \
  && pip install google-cloud-storage

# TODO: Provide environment variables here if your use case needs it 


# Change the {cloud_storage_here} part
ENTRYPOINT mlflow server --backend-store-uri "sqlite:///test.db" --default-artifact-root {cloud_storage_here} --host 0.0.0.0 --gunicorn-opts "--timeout 300"