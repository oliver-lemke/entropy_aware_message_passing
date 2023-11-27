from models.basic_gcn import BasicGCN
from utils.config import Config

import torch
import torch_geometric as tg

config = Config()


class FreeEnergyGCN(BasicGCN):
    def __init__(self, *args, **kwargs):
        """Entropic Wrapper"""
        super().__init__(*args, **kwargs)

        params = config["model_parameters"]["entropic_gcn"]

        self.T = params["temperature"]
        self.A = None
        self.weight = params["weight"]

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

        embedding = super().forward(data)

        return embedding - self.weight * self.gradient_free_energy(embedding)

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

        return torch.sum(self.dirichlet_energy(X))

    def dirichlet_energy(self, X):
        """Comute Dirichlet Energie for graph embedding X

        Returns:
            torch.tensor, shape N: Dirichlet Energies for each node
        """

        res1 = torch.einsum("ij,ik,ik->i", self.A, X, X)
        res2 = torch.einsum("ij,ik,jk->i", self.A, X, X)
        res3 = torch.einsum("ij,jk,jk->i", self.A, X, X)

        energies = 1 / 2 * (res1 - 2 * res2 + res3)

        # print(torch.sum(energies < 0))
        # print(energies[torch.where(energies < 0)])
        # print(res1[torch.where(energies < 0)])
        # print(res2[torch.where(energies < 0)])
        # print(res3[torch.where(energies < 0)])

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

        if torch.sum(~torch.isfinite(S)) > 0:
            print(torch.min(P), torch.max(P))

        return P_bar

    def gradient_dirichlet_energy(self, X):
        res1 = torch.einsum("ij,ik->ik", self.A, X)
        res2 = torch.einsum("ij,jk->ik", self.A, X)

        result = 2 * (res1 - res2)

        return result

    def gradient_free_energy(self, X):
        # TODO: check if order of magnitude match. Energy could dominate entropy
        return self.gradient_dirichlet_energy(X) - self.T * self.gradient_entropy(X)

    def gradient_entropy(self, X):
        P_bar = self.Pbar(X)
        res1 = torch.einsum("ij,ik,i->ik", self.A, X, P_bar)
        res2 = torch.einsum("ij,ik,j->ik", self.A, X, P_bar)
        res3 = torch.einsum("ij,jk,i->ik", self.A, X, P_bar)
        res4 = torch.einsum("ij,jk,j->ik", self.A, X, P_bar)

        return 1 / self.T * (res1 + res2 - res3 - res4)
