import torch

from physics.physics import Entropy


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
    result = []
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += L[i, j] * X[j] @ X[j] - 2 * A[i, j] * X[i] @ X[j]

        result.append(sum)

    result = 1 / 2 * torch.stack(result)

    result2 = []
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += (
                A[i, j] * X[i] @ X[i]
                - 2 * A[i, j] * X[i] @ X[j]
                + A[i, j] * X[j] @ X[j]
            )

        result2.append(sum)

    result2 = 1 / 2 * torch.stack(result2)

    res1 = torch.einsum("jk,ij,jk->i", X, L, X)
    res2 = torch.einsum("ik,ij,jk->i", X, A, X)
    energies = 1 / 2 * (res1 - 2 * res2)

    res1 = torch.einsum("ij,ik,ik->i", A, X, X)
    res2 = torch.einsum("ij,ik,jk->i", A, X, X)
    res3 = torch.einsum("ij,jk,jk->i", A, X, X)
    energies2 = 1 / 2 * (res1 - 2 * res2 + res3)

    assert torch.allclose(result, energies)  # E.dirichlet_energy(X)
    print("DIRICHLET ENERGY: TEST PASSED!")

    assert torch.allclose(result2, energies2)
    print("DIRICHLET ENERGY 2: TEST PASSED!")

    print(
        torch.sum(result < 0),
        torch.sum(result2 < 0),
        torch.sum(energies < 0),
        torch.sum(energies2 < 0),
    )

    assert torch.allclose(result, result2)
    print("DIRICHLET ENERGY 3: TEST PASSED!")

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
