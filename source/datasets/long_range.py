from datasets.base import BaseDataset
from torch_geometric.datasets import LRGBDataset
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class LongRangeDataset(BaseDataset, LRGBDataset):
    def __init__(self, transform):
        self.transform = transform

        root = config.get_subpath("data")
        long_range_params = config["dataset_parameters"]["long_range"]
        logger.info(f"Instantiating Long Range dataset with {root=}, {long_range_params}")
        LRGBDataset.__init__(self, root=root, **long_range_params)

    def get_input_output_dim(self) -> (int, int):
        return self.num_node_features, self.num_classes
