from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn

from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()

fusion_block_args = config["transform_block_args"]


class TransformBlock(nn.Module):
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError()


class Full(TransformBlock):
    """
    Variable fully-connected neural network.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        params = fusion_block_args[self.get_name()]
        depth = params["depth"]
        hidden_dim = params["hidden_dim"]
        self.residual = params["residual"]

        assert depth >= 2

        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)]
        )
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
            out = self.act(out)
            out = self.norm(out)
        out = self.output_layer(out)

        if self.residual:
            out = out + x

        return out

    @staticmethod
    def get_name() -> str:
        return "full"


class Id(TransformBlock):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        logger.debug(f"Unused arguments {input_dim=}, {output_dim=}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def get_name() -> str:
        return "id"


BLOCK_DICT = {Class.get_name(): Class for Class in (Full, Id)}
