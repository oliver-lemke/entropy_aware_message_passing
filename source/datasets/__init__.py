# https://github.com/LingxiaoShawn/PairNorm/blob/master/data.py
from datasets import faust, mnist, planetoid
from datasets import faust, long_range, mnist, planetoid
from datasets.base import BaseDataset
from datasets.data import load_data
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class DatasetFactory:
    def __init__(self):
        self.datasets = {
            "planetoid": planetoid.PlanetoidDataset,
            "faust": faust.FaustDataset,
            "mnist": mnist.MNISTDataset,
            "long_range": long_range.LongRangeDataset,
        }

    def get_dataset(self, *args, **kwargs) -> BaseDataset:
        dataset_name = config["dataset"]
        Dataset = self.datasets[dataset_name]
        dataset = Dataset(*args, **kwargs)

        if config["model_type"] == "pairnorm_gcn":
            dataset_data = dataset.data
            dataset_data = load_data(dataset_data)
            dataset.data = dataset_data

        return dataset

    @staticmethod
    def get_transform(transform_config):  # pylint: disable=unused-argument
        return None
