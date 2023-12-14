from datasets.base import BaseDataset
from torch_geometric.datasets import LRGBDataset
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class LongRangeDataset:
    def __init__(self, transform):
        self.transform = transform

        root = config.get_subpath("data")
        long_range_params = config["dataset_parameters"]["long_range"]
        logger.info(
            f"Instantiating Long Range dataset with {root=}, {long_range_params}"
        )
        self.train = LRGBDataset(root=root, split="train", **long_range_params)
        self.test = LRGBDataset(root=root, split="test", **long_range_params)
        self.val = LRGBDataset(root=root, split="val", **long_range_params)

    def get_input_output_dim(self) -> (int, int):
        return self.train.num_node_features, self.train.num_classes
