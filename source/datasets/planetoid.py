from datasets.base import BaseDataset
from torch_geometric.datasets import Planetoid
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class PlanetoidDataset(BaseDataset, Planetoid):
    def __init__(self, transform):
        self.transform = transform

        root = config.get_subpath("data")
        planetoid_params = config["dataset_parameters"]["planetoid"]
        logger.info(f"Instantiating Planetoid dataset with {root=}, {planetoid_params}")
        Planetoid.__init__(self, root=root, **planetoid_params)

    def get_input_output_dim(self) -> (int, int):
        return self.num_node_features, self.num_classes
