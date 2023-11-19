import torch
from physics import Entropy


def genererate_random_graph(n, d):
    """
    N: number of nodes
    D: dimension of the embedding
    """

    # compute random binary adjacency matrix
    A = torch.randint(2, size=(n, n), dtype=torch.float32)
    
    # compute degree matrix
    D = torch.diag(torch.sum(A, dim=0))

    # compute L
    L = D + A

    # same with numpy
    X = torch.randn(n, d)

    return A, L, X


def test_physics():
    """
    Test whether the dirichlet energy is computed correctly using einsum
    """
    n = 100
    d = 7
    A, L, X = genererate_random_graph(n, d)
    T = 1

    E = Entropy(T, A=A)

    # test dirichlet energy
    res = []
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += L[i, j] * X[j] @ X[j] - 2 * A[i, j] * X[i] @ X[j]

        res.append(sum)

    res = 1/2*torch.stack(res)

    assert torch.allclose(res, E.dirichlet_energy(X))
    print("DIRICHLET ENERGY: TEST PASSED!")

    # test entropy gradient
    res = []
    for i in range(n):
        sum = 0
        for j in range(n):
            contrib = (
                A[i, j] * X[i] * E.Pbar(X)[i]
                + A[i, j] * X[i] * E.Pbar(X)[j]
                - A[i, j] * X[j] * E.Pbar(X)[i]
                - A[i, j] * X[j] * E.Pbar(X)[j]
            )
            sum += contrib
        res.append(sum)

    res = torch.stack(res)

    assert torch.allclose(res, E.gradient_entropy(X))
    print("ENTROPY GRADIENT: TEST PASSED!")


test_physics()
