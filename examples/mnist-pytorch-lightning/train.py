#
# Trains an MNIST digit recognizer using PyTorch Lightning,
# and uses Mlflow to log metrics, params and artifacts

from audioop import avg
import pytorch_lightning as pl
import mlflow.pytorch
import mlflow
import logging
import os
import torch
from argparse import ArgumentParser
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utility.utility import auto_compare_and_register

try:
    from torchmetrics.functional import accuracy
except ImportError:
    # from pytorch_lightning.metrics.functional import accuracy
    pass


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super(MNISTDataModule, self).__init__()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.args = kwargs

        # transforms for images
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """

        self.df_train = datasets.MNIST(
            "dataset", download=True, train=True, transform=self.transform
        )
        self.df_train, self.df_val = random_split(self.df_train, [55000, 5000])
        self.df_test = datasets.MNIST(
            "dataset", download=True, train=False, transform=self.transform
        )

    def create_data_loader(self, df):
        """
        Generic data loader function

        :param df: Input tensor

        :return: Returns the constructed dataloader
        """
        return DataLoader(
            df, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"]
        )

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return self.create_data_loader(self.df_train)

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        return self.create_data_loader(self.df_val)

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        return self.create_data_loader(self.df_test)


class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Initializes the network
        """
        super(LightningMNISTClassifier, self).__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.optimizer = None
        self.scheduler = None
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        self.args = kwargs

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
            metavar="N",
            help="input batch size for training (default: 64)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=3,
            metavar="N",
            help="number of workers (default: 3)",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            metavar="LR",
            help="learning rate (default: 0.001)",
        )
        return parser

    def forward(self, x):
        """
        :param x: Input data

        :return: output - mnist digit label for the input image
        """
        batch_size = x.size()[0]

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param train_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches

        :param val_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - valid step loss
        """
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {"val_step_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy

        :param outputs: outputs after every epoch end

        :return: output - average valid loss
        """
        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """
        x, y = test_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        test_acc = accuracy(y_hat.cpu(), y.cpu())
        return {"test_acc": test_acc}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score

        :param outputs: outputs after every epoch end

        :return: output - average test loss
        """
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_acc", avg_test_acc)

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]


if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Autolog Mnist Example")

    # Early stopping parameters
    parser.add_argument(
        "--es_monitor", type=str, default="val_loss", help="Early stopping monitor parameter"
    )

    parser.add_argument("--es_mode", type=str, default="min", help="Early stopping mode parameter")

    parser.add_argument(
        "--es_verbose", type=bool, default=True, help="Early stopping verbose parameter"
    )

    parser.add_argument(
        "--es_patience", type=int, default=3, help="Early stopping patience parameter"
    )
    parser.add_argument(
        "--model_name", required=True, help="Nama model untuk diregistrasi."
        )
    parser.add_argument(
        "--experiment_name", required=True, help="Digunakan untuk separate dengan project / repo yang lain."
        )
    parser.add_argument(
        "--lower", required=False, help="Default True. Diguankan untuk perbandingan metric yang diinginkan\n Jika True, maka akan dilakukan registrasi model bila metric lebih rendah daripada metric di registered model"
        )
    parser.add_argument(
        "--p_metric", required=True, help="Nama metric yang akan dibandingkan."
        )
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = LightningMNISTClassifier.add_model_specific_args(parent_parser=parser)

    args = parser.parse_args()
    dict_args = vars(args)

    for _k in dict_args.keys():
        _v = dict_args[_k]
        if _v is not None:
            mlflow.log_param(_k, _v)

    if "strategy" in dict_args:
        if dict_args["strategy"] == "None":
            dict_args["strategy"] = None

    model = LightningMNISTClassifier(**dict_args)

    dm = MNISTDataModule(**dict_args)
    dm.setup(stage="fit")

    early_stopping = EarlyStopping(
        monitor=dict_args["es_monitor"],
        mode=dict_args["es_mode"],
        verbose=dict_args["es_verbose"],
        patience=dict_args["es_patience"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[lr_logger, early_stopping, checkpoint_callback], checkpoint_callback=True
    )

    # It is safe to use `mlflow.pytorch.autolog` in DDP training, as below condition invokes
    # autolog with only rank 0 gpu.

    # For CPU Training
    if dict_args["gpus"] is None or int(dict_args["gpus"]) == 0:
        mlflow.pytorch.autolog(log_models=False)
    elif int(dict_args["gpus"]) >= 1 and trainer.global_rank == 0:
        # In case of multi gpu training, the training script is invoked multiple times,
        # The following condition is needed to avoid multiple copies of mlflow runs.
        # When one or more gpus are used for training, it is enough to save
        # the model and its parameters using rank 0 gpu.
        mlflow.pytorch.autolog(log_models=False)
    else:
        # This condition is met only for multi-gpu training when the global rank is non zero.
        # Since the parameters are already logged using global rank 0 gpu, it is safe to ignore
        # this condition.
        logging.info("Active run exists.. ")

    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    
    act_run = mlflow.active_run()
    run_id = act_run.info.run_id


    # auto register model
    client = mlflow.tracking.MlflowClient()
    p_metric = dict_args["p_metric"]
    model_name = dict_args["model_name"]
    lower = dict_args["lower"]
    eval_metric = client.get_metric_history(run_id, p_metric)[0].value
    
    auto_compare_and_register(
        model=model,
        eval_metric=eval_metric,
        model_name=model_name,
        lower=lower,
        p_metric=p_metric,
        client=client,
        mlflow_model=mlflow.pytorch
        )