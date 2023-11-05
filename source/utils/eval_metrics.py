from torch.nn import functional as F


def metrics(
    pred, target, *args, reduction="sum", **kwargs  # pylint: disable=unused-argument
):
    cross_entropy = F.cross_entropy(pred, target, reduction=reduction).item()
    return {"cross_entropy": cross_entropy}
