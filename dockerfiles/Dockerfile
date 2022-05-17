FROM continuumio/miniconda3:latest

WORKDIR /mlflow/

# For now, this image only stores artifacts and metrics to the filesystem
# but in the future this is expected to change, so it should be good to
# leave this part alone.
RUN pip install mlflow>=1.0 \
  && pip install psycopg2-binary \
  && pip install azure-storage-blob==12.3.0 \
  && pip install numpy==1.21.2 \
  && pip install scipy \
  && pip install pandas==1.3.3 \
  && pip install scikit-learn==0.24.2 \
  && pip install cloudpickle

ENTRYPOINT mlflow server --host 0.0.0.0