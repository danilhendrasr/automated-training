# How to run this projek
1. Attach to this container named `timc-retrain`.
2. Upload new dataset to `new_dataset` directory (`ann_train`, `ann_val`, `dataset_train`, `dataset_val`). See `all_dataset` directory for reference. <br>
For example, you can use `new_dataset_example` directory, just make sure that you copy that and not rename it.
```
rm -r new_dataset
cp -r new_dataset_example new_dataset
```
3. Run `run.py` script.
```
python3 run.py
```

# How the program work
1. Latest model version will be copied to the next version, we will be use this new version (the copy) to "retrain" it.
2. New version will be "retrain".
3. After retrain we got the benchmark (`model-metadata.json`) for that spesific version.
4. The program will grep all benchmark from all version and save it to `benchmark.json`.
5. New dataset that we used for "retrain" will be backed up to `all_dataset` directory with the spesific version.

# How i run this docker container
```
docker run -itd --gpus all -v /home/azureuser/Projects/timc-retrain:/timc-retrain -w /timc-retrain --name timc-retrain python bash
```