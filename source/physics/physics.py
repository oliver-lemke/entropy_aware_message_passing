from __future__ import annotations

import torch

import torch_geometric as tg


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
        # self.A.requires_grad_(False)

        # given A, compute L
        # self.D = torch.sum(A, dim=0)
        # self.D.requires_grad_(False)

        self.D = tg.utils.degree(self.A[0])

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

        # compute normalization
        norm = 1 / torch.sqrt(self.D * dim + 1e-15)

        return norm.unsqueeze(-1)

    def dirichlet_energy(self, X):
        """Comute Dirichlet Energie for graph embedding X

        Returns:
            torch.tensor, shape N: Dirichlet Energies for each node
        """

        row, col = self.A  # row holds source nodes, col holds target nodes

        # Gather X for source and target nodes
        X_i = X[row]
        X_j = X[col]

        # Compute the L2 norm squared of the differences
        diff = X_i - X_j
        l2_norms_squared = 1 / 2 * (diff**2).sum(dim=-1)

        # Initialize the result tensor with zeros for each node
        energies = torch.zeros(X.size(0), device=X.device)

        # Use index_add to sum the L2 norms for each node
        energies = energies.index_add(0, row, l2_norms_squared)

        if self.norm_energy:
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
            distribution = torch.softmax(-energies / temperature, dim=0)
        else:
            distribution = torch.exp(-energies / temperature) + 1e-10

        return distribution + 1e-10

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

        # self.A is the edge index in COO format
        row, col = self.A

        # Gather P and X for source and target nodes
        P_i = P_bar[row]
        P_j = P_bar[col]
        X_i = X[row]
        X_j = X[col]

        # Compute the term (P_j + P_i)
        P_sum = P_i + P_j

        # Compute the term (X_i - X_j)
        X_diff = X_i - X_j

        # Multiply the terms
        term = P_sum[:, None] * X_diff

        # Initialize the result tensor
        result = torch.zeros_like(X)

        # Use scatter_add to sum the contributions for each node
        result = result.index_add_(0, row, term)

        if scale_by_temperature:
            result = (1 / temperature) * result

        if self.norm_energy:
            result = result * self.energy_normalization(X.shape[1])

        return result
