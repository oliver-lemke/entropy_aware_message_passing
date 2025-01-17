import torch
from torch import nn

import torch_geometric as tg
from physics import physics
from torch_geometric import nn as tnn
from utils.config import Config

config = Config()


class EntropicLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gcn_conv = tnn.GCNConv(input_dim, output_dim)
        self.params = config["model_parameters"]["entropic_gcn"]
        self.scale_by_temperature = self.params["scale_by_temperature"]

    def forward(self, x, edge_index, weight, temperature, entropy):
        x = self.gcn_conv(x, edge_index)
        # with torch.no_grad():
        entropy_gradient = entropy.gradient_entropy(
            x, temperature, scale_by_temperature=self.scale_by_temperature
        )
        x = x + weight * entropy_gradient
        return x


class EntropicGCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """Entropic Wrapper"""
        super().__init__()

        self.params = config["model_parameters"]["entropic_gcn"]
        depth = self.params["depth"]
        self.hidden_dim = self.params["hidden_dim"]
        self.convs = nn.ModuleList(
            [EntropicLayer(input_dim, self.hidden_dim)]
            + [
                EntropicLayer(self.hidden_dim, self.hidden_dim)
                for _ in range(depth - 1)
            ]
        )
        self.conv_out = tnn.GCNConv(self.hidden_dim, output_dim)
        self.relu = nn.ReLU()
        # self.norm = nn.LayerNorm(self.hidden_dim)

        temperature_params = self.params["temperature"]
        temperature_value = temperature_params["value"]
        if temperature_params["learnable"]:
            self.temperature = nn.Parameter(torch.tensor(temperature_value))
        else:
            self.temperature = temperature_value

        weight_params = self.params["weight"]
        weight_value = weight_params["value"]
        if weight_params["learnable"]:
            self.weight = nn.Parameter(torch.tensor(weight_value))
        else:
            self.weight = weight_value

        self.norm_energy = self.params["normalize_energies"]
        self.norm_dist = self.params["normalize_distribution"]

        self.A = None
        self.entropy = None

        # TODO
        # 1. switch to efficient (sparse?) framework
        # 2. once implemented, run tests whether this actually works
        # 3. in particular, check if energies are positive
        # 4. maybe track the energy during training as a metric?!

    def forward(self, data):
        """Adjust forward pass to include gradient ascend on the entropy

        Args:
            data (_type_): _description_
        """

        self.entropy = physics.Entropy(
            data.edge_index, norm_energy=self.norm_energy, norm_dist=self.norm_dist
        )

        x, edge_index = data.x, data.edge_index
        intermediate_representations = {}  # {0: x}
        idx = 1

        # First Graph Convolution
        for conv in self.convs:
            x = conv(
                x,
                edge_index,
                self.weight,
                self.temperature,
                self.entropy,
            )
            x = self.relu(x)
            # x = self.norm(x)
            intermediate_representations[idx] = x
            idx += 1

        # Second Graph Convolution
        embedding = self.conv_out(x, edge_index)
        intermediate_representations["final"] = embedding

        # log parameters
        if self.params["temperature"]["learnable"]:
            temperature_data = self.temperature.data
        else:
            temperature_data = self.temperature

        if self.params["weight"]["learnable"]:
            weight_data = self.weight.data
        else:
            weight_data = self.weight

        return (
            embedding,
            intermediate_representations,
            {
                "parameters/temperature": temperature_data,
                "parameters/weight": weight_data,
            },
        )

    def clamp_learnables(self):
        if self.params["temperature"]["learnable"]:
            self.temperature.data = torch.clamp(self.temperature, min=1e-5)
        if self.params["weight"]["learnable"]:
            self.weight.data = torch.clamp(self.weight, min=1e-5)
