import torch
from torch import nn

import torch_geometric as tg
from torch_geometric import nn as tnn
from utils.config import Config

config = Config()


class EntropicGCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """Entropic Wrapper"""
        super().__init__()

        self.params = config["model_parameters"]["entropic_gcn"]
        depth = self.params["depth"]
        self.hidden_dim = self.params["hidden_dim"]
        self.convs = nn.ModuleList(
            [tnn.GCNConv(input_dim, self.hidden_dim)]
            + [tnn.GCNConv(self.hidden_dim, self.hidden_dim) for _ in range(depth - 1)]
        )
        self.conv_out = tnn.GCNConv(self.hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(self.hidden_dim)

        temperature_params = self.params["temperature"]
        temperature_value = temperature_params["value"]
        if temperature_params["learnable"]:
            self.temperature = nn.Parameter(torch.tensor([temperature_value]))
        else:
            self.temperature = temperature_value

        weight_params = self.params["weight"]
        weight_value = weight_params["value"]
        if weight_params["learnable"]:
            self.weight = nn.Parameter(torch.tensor([weight_value]))
        else:
            self.weight = weight_value

        self.normalize_energies = self.params["normalize_energies"]

        self.A = None
        self.energy_normalization = None

        # TODO
        # 1. switch to efficient (sparse?) framework
        # 2. once implemented, run tests whether this actually works
        # 3. in particular, check if energies are potitive
        # 4. maybe track the energy during training as a metric?!

    def forward(self, data):
        """Adjust forward pass to include gradient ascend on the entropy

        Args:
            data (_type_): _description_
        """

        if self.A is None:
            self.A = tg.utils.to_dense_adj(data.edge_index).squeeze()
            self.energy_normalization = self.compute_energy_normalization()

        x, edge_index = data.x, data.edge_index
        intermediate_representations = {}  # {0: x}
        idx = 1

        # First Graph Convolution
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.norm(x)
            intermediate_representations[idx] = x
            idx += 1

        # Second Graph Convolution
        embedding = self.conv_out(x, edge_index)
        intermediate_representations["final"] = embedding

        return (
            embedding + self.weight * self.gradient_entropy(embedding),
            intermediate_representations,
        )

    def entropy(self, X):
        """Compute the entropy of the graph embedding X

        Returns:
            float: entropy of X
        """

        distribution = self.boltzmann_distribution(X)

        S = -torch.sum(distribution * torch.log(distribution))

        return S

    def total_dirichlet_energy(self, X):
        """Compute the total dirichlet energy of the graph embedding X

        Args:
            X (torch.tensor, shape NxN): graph embedding
        """

        return torch.mean(self.dirichlet_energy(X))

    def compute_energy_normalization(self):
        """
        Compute normalization for each nodes dirichlet energy.
        """

        # compute degree of each node
        degrees = torch.sum(self.A, dim=1)

        # compute normalization
        normalization = 1 / torch.sqrt(degrees * self.hidden_dim)

        return normalization

    def dirichlet_energy(self, X):
        """Comute Dirichlet Energie for graph embedding X

        Returns:
            torch.tensor, shape N: Dirichlet Energies for each node
        """

        res1 = torch.einsum("ij,ik,ik->i", self.A, X, X)
        res2 = torch.einsum("ij,ik,jk->i", self.A, X, X)
        res3 = torch.einsum("ij,jk,jk->i", self.A, X, X)

        energies = 1 / 2 * (res1 - 2 * res2 + res3)

        if self.normalize_energies:
            energies = energies * self.energy_normalization

        # (because of numerical errors ??) some energies are an epsilon negative. Clamp those.
        # FIXME still, not sure why this is happening. We should see whether this issue goes away
        # with sparse matrices
        energies.clamp(min=1e-10)

        # print(f"energies: {torch.sum(~torch.isfinite(energies))}")

        return energies

    def boltzmann_distribution(self, X):
        """Compute Boltzmann distribution

        Returns:
            torch.tensor, shape N: Boltzmann distribution given energies and temperature
        """

        energies = self.dirichlet_energy(X)

        # return softmax of energies scaled by temperature
        # adding an epsilon is a hack to avoid log(0) = -inf.
        # x*ln(x) goes to 0 as x goes to 0, so this is okay
        distribution = torch.softmax(-energies / self.temperature, dim=0) + 1e-10

        return distribution

    def Pbar(self, X):
        P = self.boltzmann_distribution(X)
        S = self.entropy(X)
        P_bar = P * (S + torch.log(P))

        return P_bar

    def gradient_entropy(self, X):
        P_bar = self.Pbar(X)
        res1 = torch.einsum("ij,ik,i->ik", self.A, X, P_bar)
        res2 = torch.einsum("ij,ik,j->ik", self.A, X, P_bar)
        res3 = torch.einsum("ij,jk,i->ik", self.A, X, P_bar)
        res4 = torch.einsum("ij,jk,j->ik", self.A, X, P_bar)

        result = 1 / self.temperature * (res1 + res2 - res3 - res4)

        if self.normalize_energies:
            result = result * self.energy_normalization[:, None]

        """
        print(result)
        test = self.Pbar(X)

        res = []
        for i in range(X.shape[0]):
            sum = 0
            for j in range(X.shape[0]):
                contrib = self.energy_normalization[i]*(
                    self.A[i, j] * X[i] * test[i]
                    + self.A[i, j] * X[i] * test[j]
                    - self.A[i, j] * X[j] * test[i]
                    - self.A[i, j] * X[j] * test[j]
                )
                sum += contrib
            res.append(sum)

            print(1 / self.temperature * sum)

        """

        return result

    def clamp_learnables(self):
        if self.params["temperature"]["learnable"]:
            self.temperature = torch.clamp(self.temperature, min=1e-5)
        if self.params["weight"]["learnable"]:
            self.weight = torch.clamp(self.weight, min=1e-5)
