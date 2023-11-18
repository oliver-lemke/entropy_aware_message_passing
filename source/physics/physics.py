import numpy as np
from scipy.special import softmax


class Entropy:
    def __init__(self, A, X, T):
        """ Class to handle the entropy of a given Graph

        Args:
            A (np.array, shape NxN): Adjacency matrix
            X (np.array, shape NxD): Graph embedding
            T (float): Temperature
        """
        self.A = A
        self.X = X
        self.T = T

        # given A, compute L
        D = np.diag(np.sum(A, axis=0))
        self.L = D + A


    def update(self, X):
        self.X = X

    def entropy(self):
        """Compute the entropy of the graph embedding X

        Returns:
            float: entropy of X
        """
        energies = self.dirichlet_energy()
        distribution = self.boltzmann_distribution()
        S = -np.sum(distribution * np.log(distribution))

        return S

    def dirichlet_energy(self):
        """Comute Dirichlet Energie for graph embedding X

        Returns:
            np.array, shape N: Dirichlet Energies for each node
        """

        # for every row in X, compute the inner product with respect to L. Stack all of the inner products into a vector

        res1 = np.einsum("jk,ij,jk->i", self.X, self.L, self.X)
        res2 = np.einsum("ik,ij,jk->i", self.X, self.A, self.X)

        energies = res1 - 2 * res2

        return energies

    def boltzmann_distribution(self):
        """Compute Boltzmann distribution

        Returns:
            np.array, shape N: Boltzmann distribution given energies and temperature
        """

        energies = self.dirichlet_energy()

        return softmax(-energies / self.T)

    def Pbar(self):
        P = self.boltzmann_distribution()
        S = self.entropy()
        P_bar = P * (S + np.log(P))

        return P_bar

    def gradient_entropy(self):
        P_bar = self.Pbar()
        res1 = np.einsum("ij,ik,i->ik", self.A, self.X, P_bar)
        res2 = np.einsum("ij,ik,j->ik", self.A, self.X, P_bar)
        res3 = np.einsum("ij,jk,i->ik", self.A, self.X, P_bar)
        res4 = np.einsum("ij,jk,j->ik", self.A, self.X, P_bar)

        grad = 1 / self.T * (res1 + res2 - res3 - res4)

        return grad
