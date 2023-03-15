import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from hydra.utils import instantiate
from torchmetrics.classification import MulticlassROC


class Net(pl.LightningModule):
    def __init__(self, model, criterion, num_classes, optimizer, scheduler=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.confusion_matrix = tm.ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )
        self.roc = MulticlassROC(num_classes=num_classes)

        self.train_accuracy = tm.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = tm.Accuracy(task="multiclass", num_classes=num_classes)

    def configure_optimizers(self):
        if self.scheduler:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": instantiate(self.scheduler, optimizer=self.optimizer),
                "monitor": "val_loss",
            }
        return self.optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)

        self.train_accuracy.update(y_hat, y) 

        return loss
    
    def training_epoch_end(self, outs):
        self.log("train_accuracy", self.train_accuracy.compute())
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)

        self.confusion_matrix.update(y_hat, y)
        self.roc.update(y_hat, y)
        self.val_accuracy.update(y_hat, y)
        
        return loss

    def validation_epoch_end(self, outs):
        self.log("val_accuracy", self.val_accuracy.compute())
        self.val_accuracy.reset()

        cm = self.confusion_matrix.compute()
        roc = self.roc.compute()

        self.confusion_matrix.reset()
        self.roc.reset()

        if self.logger:
            run_name = self.logger.name + "/" + str(self.logger.version)
        else:
            run_name = "default"
        
        os.makedirs(f"confusion_matrices/{run_name}", exist_ok=True)
        os.makedirs(f"rocs/{run_name}", exist_ok=True)

        torch.save(
            cm, os.path.join(f"confusion_matrices/{run_name}", "confusion_matrix.pt")
        )
        torch.save(
            roc, os.path.join(f"rocs/{run_name}", "roc.pt")
        )
