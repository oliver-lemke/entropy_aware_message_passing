from __future__ import annotations

import torch


class Entropy:
    def __init__(self, A=0, norm_energy=False, norm_dist=False):
        """Class to handle the entropy of a given Graph

        Args:
            A (torch.tensor, shape NxN): Adjacency matrix
            X (torch.tensor, shape NxD): Graph embedding
            T (float): Temperature
        """

        # TODO: implement everything with sparse matrices. Don't use adjacency matrix, but edge_index
        self.A = None
        self.D = None

        self.norm_energy = norm_energy
        self.norm_dist = norm_dist

        # adjacency matrix
        self.update_adj(A)

    def update_adj(self, A):
        """Update the adjacency matrix

        Args:
            A (torch.tensor, shape NxN): Adjacency matrix
        """

        self.A = A

        # given A, compute L
        self.D = torch.diag(torch.sum(A, dim=0))

    def entropy(self, X, temperature):
        """Compute the entropy of the graph embedding X

        Returns:
            float: entropy of X
        """

        distribution = self.boltzmann_distribution(X, temperature)

        S = -torch.sum(distribution * torch.log(distribution))

        return S

    def total_dirichlet_energy(self, X):
        """Compute the total dirichlet energy of the graph embedding X

        Args:
            X (torch.tensor, shape NxN): graph embedding
        """

        return torch.mean(self.dirichlet_energy(X))

    def energy_normalization(self, dim: int):
        """
        Compute normalization for each nodes dirichlet energy.
        """

        # compute degree of each node
        # degrees = torch.sum(self.A, dim=1)
        degrees = torch.diagonal(self.D, 0)

        # compute normalization
        norm = 1 / torch.sqrt(degrees * dim)

        return norm.unsqueeze(-1)

    def dirichlet_energy(self, X):
        """Comute Dirichlet Energie for graph embedding X

        Returns:
            torch.tensor, shape N: Dirichlet Energies for each node
        """

        res1 = torch.einsum("ij,ik,ik->i", self.A, X, X)
        res2 = torch.einsum("ij,ik,jk->i", self.A, X, X)
        res3 = torch.einsum("ij,jk,jk->i", self.A, X, X)

        energies = 1 / 2 * (res1 - 2 * res2 + res3)

        if self.norm_energy:
            # energies = (
            #    energies * self.energy_normalization(X.shape[1]).squeeze()
            # )
            energies *= self.energy_normalization(X.shape[1]).squeeze()

        # (because of numerical errors ??) some energies are an epsilon negative. Clamp those.
        # FIXME still, not sure why this is happening. We should see whether this issue goes away
        # with sparse matrices
        energies.clamp(min=1e-10)

        return energies

    def boltzmann_distribution(self, X, temperature):
        """Compute Boltzmann distribution

        Returns:
            torch.tensor, shape N: Boltzmann distribution given energies and temperature
        """

        energies = self.dirichlet_energy(X)

        # return softmax of energies scaled by temperature
        # adding an epsilon is a hack to avoid log(0) = -inf.
        # x*ln(x) goes to 0 as x goes to 0, so this is okay
        if self.norm_dist:
            distribution = torch.softmax(-energies / temperature, dim=0) + 1e-10
        else:
            distribution = torch.exp(-energies / temperature) + 1e-10

        return distribution

    def Pbar(self, X, temperature):
        P = self.boltzmann_distribution(X, temperature)

        if self.norm_dist:
            S = self.entropy(X, temperature)
            P_bar = P * (S + torch.log(P))
        else:
            P_bar = P * (1 + torch.log(P))

        return P_bar

    def gradient_entropy(
        self, X, temperature: float, scale_by_temperature: bool = False
    ):
        P_bar = self.Pbar(X, temperature)
        res1 = torch.einsum("ij,ik,i->ik", self.A, X, P_bar)
        res2 = torch.einsum("ij,ik,j->ik", self.A, X, P_bar)
        res3 = torch.einsum("ij,jk,i->ik", self.A, X, P_bar)
        res4 = torch.einsum("ij,jk,j->ik", self.A, X, P_bar)

        result = res1 + res2 - res3 - res4
        if scale_by_temperature:
            result = (1 / temperature) * result

        if self.norm_energy:
            result = result * self.energy_normalization(X.shape[1])

        return result
