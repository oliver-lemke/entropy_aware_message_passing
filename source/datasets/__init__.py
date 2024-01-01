from datasets import faust, mnist, planetoid
from datasets.base import BaseDataset
from utils.config import Config
from utils.logs import Logger
from datasets.data import load_data

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
        data = Dataset(*args, **kwargs)
        if config["model_type"] == "pairnorm_gcn":
            data = load_data(data)
        return data

    @staticmethod
    def get_transform(transform_config):  # pylint: disable=unused-argument
        return None
