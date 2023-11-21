import torch


class Entropy:
    def __init__(self, T, A=0):
        """ Class to handle the entropy of a given Graph

        Args:
            A (torch.tensor, shape NxN): Adjacency matrix
            X (torch.tensor, shape NxD): Graph embedding
            T (float): Temperature
        """

        # TODO: implement everything with sparse matrices. Don't use adjacency matrix, but edge_index

        # adjacency matrix
        self.update_adj(A)

        # temperature
        self.T = T

    def update_adj(self, A):
        """Update the adjacency matrix

        Args:
            A (torch.tensor, shape NxN): Adjacency matrix
        """

        self.A = A

        # given A, compute L
        D = torch.diag(torch.sum(A, dim=0))
        self.L = D + A

    def entropy(self, X):
        """Compute the entropy of the graph embedding X

        Returns:
            float: entropy of X
        """

        distribution = self.boltzmann_distribution(X)

        S = -torch.sum(distribution * torch.log(distribution))

        return S

    def dirichlet_energy(self, X):
        """Comute Dirichlet Energie for graph embedding X

        Returns:
            torch.tensor, shape N: Dirichlet Energies for each node
        """

        res1 = torch.einsum("ij,ik,ik->i", self.A, X, X)
        res2 = torch.einsum("ij,ik,jk->i", self.A, X, X)
        res3 = torch.einsum("ij,jk,jk->i", self.A, X, X)

        energies = 1/2*(res1 - 2 * res2 + res3)

        # FIXME there are sometimes negative energies, which is mathematically impossible?!
        # But, we should use sparse matrices anyways. Maybe then, if we just do ordinary matrix
        # multiplication, we don't have this problem anymore?
        # also, values are complete garbage
        energies.clamp(min=1e-10)

        #res1 = torch.einsum("jk,ij,jk->i", X, self.L, X)
        #res2 = torch.einsum("ik,ij,jk->i", X, self.A, X)
        #energies = 1/2*(res1 - 2 * res2)

        # abbreviate this for loop using torch.einsum
        return energies

    

    def boltzmann_distribution(self, X):
        """Compute Boltzmann distribution

        Returns:
            torch.tensor, shape N: Boltzmann distribution given energies and temperature
        """

        # FIXME: right now, normalize with shape s.t. energies are non-zero
        energies = self.dirichlet_energy(X)

        # return softmax of energies scaled by temperature
        distribution = torch.softmax(-energies / self.T, dim=0)

        # this is a hack to avoid log(0) = -inf. x*ln(x) goes to 0 as x goes to 0, so this is okay
        distribution += 1e-10
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

        return 1 / self.T * (res1 + res2 - res3 - res4)
