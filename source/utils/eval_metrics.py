import torch
from torch.nn import functional as F


def dirichlet_energy(self, A, X):
    """Comute Dirichlet Energie for graph embedding X

    Returns:
        torch.tensor, shape N: Dirichlet Energies for each node
    """

    res1 = torch.einsum("ij,ik,ik->i", A, X, X)
    res2 = torch.einsum("ij,ik,jk->i", A, X, X)
    res3 = torch.einsum("ij,jk,jk->i", A, X, X)

    energies = 1 / 2 * (res1 - 2 * res2 + res3)

    # FIXME there are sometimes negative energies, which is mathematically
    #  impossible?! But, we should use sparse matrices anyways. Maybe then,
    #  if we just do ordinary matrix multiplication, we don't have this problem
    #  anymore? also, values are complete garbage
    energies.clamp(min=1e-10)

    # res1 = torch.einsum("jk,ij,jk->i", X, self.L, X)
    # res2 = torch.einsum("ik,ij,jk->i", X, self.A, X)
    # energies = 1/2*(res1 - 2 * res2)

    # abbreviate this for loop using torch.einsum
    return energies


def accuracy(pred, target, reduction="sum"):
    corrects = torch.argmax(pred, dim=1) == target
    if reduction == "sum":
        ret = torch.sum(corrects)
    elif reduction == "mean":
        ret = torch.mean(corrects)
    else:
        raise ValueError(f"Unknown reduction type {reduction}!")
    return ret


def metrics(
    pred, target, *args, reduction="sum", **kwargs  # pylint: disable=unused-argument
):
    cross_entropy = F.cross_entropy(pred, target, reduction=reduction).item()
    return {"cross_entropy": cross_entropy, "accuracy": accuracy(pred, target)}
