from torch import nn

from torch_geometric import nn as tnn
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class BasicGCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicGCN, self).__init__()
        params = config["model_parameters"]["basic_gcn"]
        depth = params["depth"]
        hidden_dim = params["hidden_dim"]

        self.convs = nn.ModuleList(
            [tnn.GCNConv(input_dim, hidden_dim)]
            + [tnn.GCNConv(hidden_dim, hidden_dim) for _ in range(depth - 1)]
        )
        self.conv_out = tnn.GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.log_softmax = nn.LogSoftmax(dim=1)
        logger.debug(str(self))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First Graph Convolution
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)

        # Second Graph Convolution
        x = self.conv_out(x, edge_index)

        return self.log_softmax(x)
