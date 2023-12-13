from __future__ import annotations

import torch


class Entropy:
    def __init__(self, A=0, normalize_energies=False):
        """Class to handle the entropy of a given Graph

        Args:
            A (torch.tensor, shape NxN): Adjacency matrix
            X (torch.tensor, shape NxD): Graph embedding
            T (float): Temperature
        """

        # TODO: implement everything with sparse matrices. Don't use adjacency matrix, but edge_index
        self.A = None
        self.L = None
        self.D = None

        self.normalize_energies = normalize_energies

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
        self.L = self.D + A

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

    def compute_energy_normalization(self, dim: int):
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

        if self.normalize_energies:
            energies = (
                energies * self.compute_energy_normalization(X.shape[1]).squeeze()
            )

        # (because of numerical errors ??) some energies are an epsilon negative. Clamp those.
        # FIXME still, not sure why this is happening. We should see whether this issue goes away
        # with sparse matrices
        energies.clamp(min=1e-10)

        # print(f"energies: {torch.sum(~torch.isfinite(energies))}")

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
        distribution = torch.softmax(-energies / temperature, dim=0) + 1e-10

        return distribution

    def Pbar(self, X, temperature):
        P = self.boltzmann_distribution(X, temperature)
        S = self.entropy(X, temperature)
        P_bar = P * (S + torch.log(P))

        return P_bar

    def gradient_entropy(self, X, temperature: float):
        P_bar = self.Pbar(X, temperature)
        res1 = torch.einsum("ij,ik,i->ik", self.A, X, P_bar)
        res2 = torch.einsum("ij,ik,j->ik", self.A, X, P_bar)
        res3 = torch.einsum("ij,jk,i->ik", self.A, X, P_bar)
        res4 = torch.einsum("ij,jk,j->ik", self.A, X, P_bar)

        result = 1 / temperature * (res1 + res2 - res3 - res4)

        if self.normalize_energies:
            result = result * self.compute_energy_normalization(X.shape[1])

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
