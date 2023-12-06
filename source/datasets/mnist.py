from datasets.base import BaseDataset
from torch_geometric.datasets import MNISTSuperpixels
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class MNISTDataset(BaseDataset, MNISTSuperpixels):
    def __init__(self, transform):
        self.transform = transform

        root = config.get_subpath("data")
        MNISTSuperpixels.__init__(self, root=root)

    def get_input_output_dim(self) -> (int, int):
        return self.num_node_features, self.num_classes
