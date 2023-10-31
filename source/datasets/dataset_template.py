"""
Explanation of the dataset
"""
import os.path

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import cv2
import pandas as pd
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class TemplateDataset(Dataset):
    def __init__(self, split: str, transformations=None, flags=None):
        """
        Explanation of the dataset
        """
        self.split = split
        self.transformations = transformations
        self.flags = flags

        if self.transformations is None:
            self.transformations = transforms.ToTensor()

        if self.flags is None:
            self.flags = {}

        file_path = self._get_file_path()
        data_path = config.build_subpath("data/data_template/data/data/")
        df = pd.read_csv(file_path)
        df["path"] = df.apply(
            lambda row: os.path.join(
                data_path,
                f"input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg",
            ),
            axis=1,
        )
        df["code"] = df["code"] - 1
        self.df = df.drop(columns=["suite_id", "sample_id", "character", "value"])
        self.length = df.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        series = self.df.iloc[idx]
        image_bgr = cv2.imread(series["path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb.astype(np.float32)
        image_tensor = self.transformations(image_rgb)
        target = series["code"]
        return {"input": image_tensor, "target": target}

    def _get_file_path(self):
        if self.split == "train":
            return config.build_subpath("data/data_template/train.csv")
        if self.split == "val":
            return config.build_subpath("data/data_template/val.csv")
        raise ValueError(f"Invalid split {self.split}")


def compute_transforms(transform_config):  # pylint: disable=unused-argument
    return transforms.ToTensor()


def main():
    pass
    # use for testing


if __name__ == "__main__":
    main()
