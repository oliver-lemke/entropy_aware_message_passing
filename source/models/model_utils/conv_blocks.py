from torch import nn

from torch_geometric import nn as tnn
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()

conv_block_args = config["conv_block_args"]


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        params = conv_block_args["basic"]
        self.conv = tnn.GCNConv(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=params["dropout_rate"])

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        return x


BLOCK_DICT = {"basic": BasicBlock}
