import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv
from torch_scatter import scatter
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class G2(nn.Module):
    def __init__(self, conv, p=2.0, conv_type="GCN", activation=nn.ReLU()):
        super(G2, self).__init__()
        self.conv = conv
        self.p = p
        self.activation = activation
        self.conv_type = conv_type

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        if self.conv_type == "GAT":
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = self.activation(self.conv(X, edge_index))
        gg = torch.tanh(
            scatter(
                (torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
                edge_index[0],
                0,
                dim_size=X.size(0),
                reduce="mean",
            )
        )

        return gg


class G2_GNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(G2_GNN, self).__init__()
        params = config["model_parameters"]["g2"]

        self.conv_type = params["conv_type"]
        nhid = params["hidden_dim"]
        drop_in = params["input_dropout"]
        drop = params["dropout"]
        p = params["p"]
        self.enc = nn.Linear(input_dim, nhid)
        self.dec = nn.Linear(nhid, output_dim)
        self.drop_in = drop_in
        self.drop = drop
        self.nlayers = params["depth"]
        if self.conv_type == "GCN":
            self.conv = GCNConv(nhid, nhid)
            if params["use_gg_conv"] == True:
                self.conv_gg = GCNConv(nhid, nhid)
        elif self.conv_type == "GAT":
            self.conv = GATConv(nhid, nhid, heads=4, concat=True)
            if params["use_gg_conv"] == True:
                self.conv_gg = GATConv(nhid, nhid, heads=4, concat=True)
        else:
            print("specified graph conv not implemented")

        if params["use_gg_conv"] == True:
            self.G2 = G2(self.conv_gg, p, self.conv_type, activation=nn.ReLU())
        else:
            self.G2 = G2(self.conv, p, self.conv_type, activation=nn.ReLU())
        logger.debug(str(self))

    def forward(self, data):
        X = data.x
        n_nodes = X.size(0)
        edge_index = data.edge_index
        X = F.dropout(X, self.drop_in, training=self.training)
        X = torch.relu(self.enc(X))
        intermediate_representations = {}  # {0: x}
        idx = 1

        for i in range(self.nlayers):
            if self.conv_type == "GAT":
                X_ = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                X_ = torch.relu(self.conv(X, edge_index))
            tau = self.G2(X, edge_index)
            X = (1 - tau) * X + tau * X_
            intermediate_representations[idx] = X
        X = F.dropout(X, self.drop, training=self.training)
        X = torch.relu(self.dec(X))

        intermediate_representations["final"] = X
        return X, intermediate_representations, {}

    def clamp_learnables(self):
        # no learnable parameters
        pass
