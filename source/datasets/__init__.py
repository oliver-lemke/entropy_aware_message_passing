from datasets import faust, mnist, planetoid
from datasets.base import BaseDataset
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
        }

    def get_dataset(self, *args, **kwargs) -> BaseDataset:
        dataset_name = config["dataset"]
        Dataset = self.datasets[dataset_name]
        return Dataset(*args, **kwargs)

    @staticmethod
    def get_transform(transform_config):  # pylint: disable=unused-argument
        return None
