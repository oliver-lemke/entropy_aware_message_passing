"""
@inproceedings{
zhao2020pairnorm,
title={PairNorm: Tackling Oversmoothing in {\{}GNN{\}}s},
author={Lingxiao Zhao and Leman Akoglu},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=rkecl1rtwB}
}
"""

from models.model_utils.layers import *
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class PairNormGCN(nn.Module):
    # source: https://github.com/LingxiaoShawn/PairNorm/blob/master/models.py
    def __init__(self, input_dim, output_dim):
        super(PairNormGCN, self).__init__()
        self.params = config["model_parameters"]["pairnorm_gcn"]
        hidden_dim = self.params["hidden_dim"]
        nlayer = self.params["depth"]
        dropout = self.params["dropout"]
        residual = self.params["residual"]
        norm_mode = str(self.params["norm_mode"])
        norm_scale = self.params["norm_scale"]

        assert nlayer >= 1
        self.hidden_layers = nn.ModuleList(
            [
                GraphConv(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(nlayer - 1)
            ]
        )
        self.out_layer = GraphConv(input_dim if nlayer == 1 else hidden_dim, output_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.relu = nn.ReLU(True)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.skip = residual

    def forward(self, data):
        x, adj = data.x, data.adj
        x_old = 0

        intermediate_representations = {}  # {0: x}
        idx = 1
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip > 0 and i % self.skip == 0:
                x = x + x_old
                x_old = x
            intermediate_representations[idx] = x
            idx += 1

        x = self.dropout(x)
        x = self.out_layer(x, adj)
        intermediate_representations["final"] = x
        return x, intermediate_representations, {}

    def clamp_learnables(self):
        # no learnable parameters
        pass
