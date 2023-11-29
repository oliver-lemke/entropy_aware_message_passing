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
        self.hidden_dim = params["hidden_dim"]

        self.conv1 = tnn.GCNConv(input_dim, self.hidden_dim)
        self.conv2 = tnn.GCNConv(self.hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.log_softmax = nn.LogSoftmax(dim=1)
        logger.debug(str(self))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First Graph Convolution
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(
            x,
        )

        # Second Graph Convolution
        x = self.conv2(x, edge_index)

        return self.log_softmax(x)
