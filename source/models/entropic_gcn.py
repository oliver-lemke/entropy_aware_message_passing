from models.basic_gcn import BasicGCN
from utils.config import Config

import torch
import torch_geometric as tg

config = Config()


class EntropicGCN(BasicGCN):
    def __init__(self, *args, **kwargs):
        """Entropic Wrapper"""
        super().__init__(*args, **kwargs)

        params = config["model_parameters"]["entropic_gcn"]

        self.T = params["temperature"]
        self.weight = params["weight"]
        self.normalize_energies = params["normalize_energies"]

        self.A = None

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

        embedding = super().forward(data)

        return embedding + self.weight * self.gradient_entropy(embedding)

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
        distribution = torch.softmax(-energies / self.T, dim=0) + 1e-10

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

        result = 1 / self.T * (res1 + res2 - res3 - res4)

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

            print(1 / self.T * sum)

        """

        return result
