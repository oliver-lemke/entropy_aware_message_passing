from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn

from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()

fusion_block_args = config["fusion_block_args"]


class FusionBlock(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, branch_tensors: dict[str, torch.Tensor], this_index: int
    ) -> torch.Tensor:
        raise NotImplementedError()


class SumFusion(FusionBlock):
    def __init__(self):
        super().__init__()

    def forward(
        self, branch_tensors: dict[str, torch.Tensor], this_index: int
    ) -> torch.Tensor:
        concat_tensors = torch.stack(list(branch_tensors.values()), dim=-1)
        return torch.sum(concat_tensors, dim=-1)


BLOCK_DICT = {"sum": SumFusion}
