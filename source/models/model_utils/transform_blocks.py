from __future__ import annotations

import torch
from torch import nn

from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()

fusion_block_args = config["transform_block_args"]


class Linear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        params = fusion_block_args["sum"]
        depth = params["depth"]
        hidden_dim = params["hidden_dim"]

        assert depth >= 2

        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 2)]
        )
        self.act = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.act
        return self.output_layer(x)


BLOCK_DICT = {"linear": Linear}
