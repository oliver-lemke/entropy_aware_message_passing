from abc import abstractmethod

from torch import nn

from torch_geometric import nn as tnn
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()

conv_block_args = config["conv_block_args"]


class TransformBlock(nn.Module):
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError()


class BasicGCNBlock(TransformBlock):
    """
    Basic GCN
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = tnn.GCNConv(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.norm(x)
        return x

    @staticmethod
    def get_name() -> str:
        return "basic_gcn"


BLOCK_DICT = {Class.get_name(): Class for Class in (BasicGCNBlock,)}
