from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, random_split

from dataloading.datagen import CustomDataGen


class OsteosarcomaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        base_path,
        num_workers=0,
        train_val_ratio=0.8,
        img_size=None,
        k=1,
        n_splits=1,
        random_state=42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.num_workers = num_workers
        self.base_path = Path(base_path)
        self.img_size = tuple(img_size) if img_size else None
        self.k = k
        self.n_splits = n_splits
        self.random_state = random_state

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def prepare_data(self):
        f = self.base_path / "ML_Features_1144.csv"
        self.df = pd.read_csv(f)
        self.filenames = self.df["image.name"]
        self.labels = self.df["classification"]

        self.filenames = self.filenames.str.replace(" -", "-")
        self.filenames = self.filenames.str.replace("- ", "-")
        self.filenames = self.filenames.str.replace(" ", "-")
        self.filenames = self.filenames + ".jpg"

        self.labels = self.labels.str.lower()
        self.labels = self.labels.replace("non-tumor", 0)
        self.labels = self.labels.replace("viable", 1)
        self.labels = self.labels.replace("non-viable-tumor", 2)
        self.labels = self.labels.replace("viable: non-viable", 2)

        assert len(self.filenames) == len(self.labels)
        assert set(self.labels) == {0, 1, 2}

        self.num_classes = len(set(self.labels))
        self.class_weights = torch.tensor(
            compute_class_weight("balanced", classes=[0, 1, 2], y=self.labels), dtype=torch.float
        )

        self.df = pd.DataFrame({"filename": self.filenames, "label": self.labels})

    def get_preprocessing_transform(self):
        transforms = nn.Sequential(
            torchvision.transforms.Normalize(self.mean, self.std),
            torchvision.transforms.Resize(self.img_size)
            if self.img_size
            else nn.Identity(),
        )
        return transforms
    
    def get_augmentation_transform(self):
        transforms = nn.Sequential(
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(20),
        )
        return transforms

    def setup(self, stage=None):
        if self.n_splits != 1:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = [k for k in kf.split(self.df)]

            self.train_subjects, self.val_subjects = self.df.iloc[splits[self.k][0]], self.df.iloc[splits[self.k][1]]
        else:
            num_subjects = len(self.df)
            num_train_subjects = int(round(num_subjects * self.train_val_ratio))
            num_val_subjects = num_subjects - num_train_subjects
            splits = num_train_subjects, num_val_subjects
            self.train_subjects, self.val_subjects = random_split(
                self.df, splits, generator=torch.Generator().manual_seed(self.random_state)  # type: ignore
            )

    def train_dataloader(self):
        return DataLoader(
            CustomDataGen(
                self.train_subjects,
                self.base_path,
                transform=torchvision.transforms.Compose(
                    [
                        self.get_preprocessing_transform(),
                        self.get_augmentation_transform(),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            CustomDataGen(
                self.val_subjects,
                self.base_path,
                transform=self.get_preprocessing_transform(),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
