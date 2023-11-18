import numpy as np

from physics import Entropy


def genererate_random_graph(n, d):
    """
    N: number of nodes
    D: dimension of the embedding
    """

    # compute random binary adjacency matrix
    A = np.random.randint(2, size=(n, n))

    # compute degree matrix
    D = np.diag(np.sum(A, axis=0))

    # compute L
    L = D + A

    # same with numpy
    X = np.random.randn(n, d)

    return A, L, X


def test_physics():
    """
    Test whether the dirichlet energy is computed correctly using einsum
    """
    n = 10
    d = 3
    A, L, X = genererate_random_graph(n, d)
    T = 1

    E = Entropy(A, X, T)

    # test dirichlet energy
    res = []
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += L[i, j] * X[j] @ X[j] - 2 * A[i, j] * X[i] @ X[j]

        res.append(sum)

    res = np.array(res)

    assert np.allclose(res, E.dirichlet_energy())
    print("DIRICHLET ENERGY: TEST PASSED!")

    # test entropy gradient
    res = []
    for i in range(n):
        sum = 0
        for j in range(n):
            contrib = (
                A[i, j] * X[i] * E.Pbar()[i]
                + A[i, j] * X[i] * E.Pbar()[j]
                - A[i, j] * X[j] * E.Pbar()[i]
                - A[i, j] * X[j] * E.Pbar()[j]
            )
            sum += contrib
        res.append(sum)

    res = np.array(res)

    assert np.allclose(res, E.gradient_entropy())
    print("ENTROPY GRADIENT: TEST PASSED!")


test_physics()
