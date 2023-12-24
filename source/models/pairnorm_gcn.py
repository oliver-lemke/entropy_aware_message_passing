from torch import nn

from torch_geometric import nn as tnn
from utils.config import Config
from utils.logs import Logger
from source.models.model_utils.layers import *

config = Config()
logger = Logger()


class PairNormGCN(nn.Module):
    # source: https://github.com/LingxiaoShawn/PairNorm/blob/master/models.py
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0, nlayer=2, residual=0,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(PairNormGCN, self).__init__()
        assert nlayer >= 1
        self.hidden_layers = nn.ModuleList([
            GraphConv(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(nlayer - 1)
        ])
        self.out_layer = GraphConv(input_dim if nlayer == 1 else hidden_dim, output_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.relu = nn.ReLU(True)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.skip = residual

    def forward(self, x, adj):
        x_old = 0
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip > 0 and i % self.skip == 0:
                x = x + x_old
                x_old = x

        x = self.dropout(x)
        x = self.out_layer(x, adj)
        return x
