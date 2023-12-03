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
    """
    Model block used for fusing node representations.
    """

    @abstractmethod
    def forward(
        self, branch_tensors: dict[str, torch.Tensor], this_index: int
    ) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError()


class SumFusion(FusionBlock):
    """
    Simply sums all inputs.
    """

    def __init__(self, *args, **kwargs):
        logger.debug(f"Unused {args=}, {kwargs=}")
        super().__init__()

    def forward(
        self, branch_tensors: dict[str, torch.Tensor], this_index: int
    ) -> torch.Tensor:
        concat_tensors = torch.stack(list(branch_tensors.values()), dim=-1)
        return torch.sum(concat_tensors, dim=-1)

    @staticmethod
    def get_name() -> str:
        return "sum"


class MeanFusion(FusionBlock):
    """
    Simply averages all inputs.
    """

    def __init__(self, *args, **kwargs):
        logger.debug(f"Unused {args=}, {kwargs=}")
        super().__init__()

    def forward(
        self, branch_tensors: dict[str, torch.Tensor], this_index: int
    ) -> torch.Tensor:
        concat_tensors = torch.stack(list(branch_tensors.values()), dim=-1)
        return torch.mean(concat_tensors, dim=-1)

    @staticmethod
    def get_name() -> str:
        return "mean"


class MaxFusion(FusionBlock):
    """
    Simply takes max at each index over all tensors.
    """

    def __init__(self, *args, **kwargs):
        logger.debug(f"Unused {args=}, {kwargs=}")
        super().__init__()

    def forward(
        self, branch_tensors: dict[str, torch.Tensor], this_index: int
    ) -> torch.Tensor:
        concat_tensors = torch.stack(list(branch_tensors.values()), dim=-1)
        return torch.max(concat_tensors, dim=-1)[0]

    @staticmethod
    def get_name() -> str:
        return "max"


class SimpleConvFusion(FusionBlock):
    """
    Combines all inputs via 1x1 convolution
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.debug(f"Unused {args=}, {kwargs=}")
        self.params = fusion_block_args[self.get_name()]
        self.residual = self.params["residual"]

        self.layers = None
        self.act = None
        self.norm = None
        self.output_layer = None
        self.has_been_initialized = False

    def _custom_init(self, nr_fusion_tensors: int):
        assert not self.has_been_initialized
        depth = self.params["depth"]
        hidden_dim = self.params["hidden_dim"]

        self.layers = nn.ModuleList(
            [nn.Linear(nr_fusion_tensors, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)]
        )
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.to(torch.device(config["device"]))
        self.has_been_initialized = True

    def forward(
        self, branch_tensors: dict[int, torch.Tensor], this_index: int
    ) -> torch.Tensor:
        if not self.has_been_initialized:
            self._custom_init(len(branch_tensors))

        assert self.has_been_initialized
        x = torch.stack(list(branch_tensors.values()), dim=-1)
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
            x = self.norm(x)
        x = self.output_layer(x)
        x = x.squeeze()

        if self.residual:
            x = x + branch_tensors[this_index]

        return x

    @staticmethod
    def get_name() -> str:
        return "simple_conv"


class SimpleAttentionFusion(FusionBlock):
    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.debug(f"Unused {args=}, {kwargs=}")
        self.params = fusion_block_args[self.get_name()]
        self.residual = self.params["residual"]
        self.attention_type = self.params["attention_type"]

        self.attention_layers = None
        self.layers = None
        self.act = None
        self.norm = None
        self.output_layer = None
        self.softmax = None
        self.has_been_initialized = False

    def _custom_init(self, dim: int):
        assert not self.has_been_initialized

        # basics
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-2)

        # attention
        attention_depth = self.params["attention_depth"]
        attention_hidden_dim = self.params["attention_hidden_dim"]
        self.attention_layers = nn.ModuleList(
            [nn.Linear(dim, attention_hidden_dim)]
            + [
                nn.Linear(attention_hidden_dim, attention_hidden_dim)
                for _ in range(attention_depth - 1)
            ]
        )
        self.attention_norm = nn.LayerNorm(attention_hidden_dim)
        if self.attention_type == "per_node":
            self.attention_output_layer = nn.Linear(attention_hidden_dim, 1)
        elif self.attention_type == "per_element":
            self.attention_output_layer = nn.Linear(attention_hidden_dim, dim)

        # normal transformation
        depth = self.params["depth"]
        hidden_dim = self.params["hidden_dim"]
        self.layers = nn.ModuleList(
            [nn.Linear(dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, dim)

        self.to(torch.device(config["device"]))
        self.has_been_initialized = True

    def forward(
        self, branch_tensors: dict[int, torch.Tensor], this_index: int
    ) -> torch.Tensor:
        if not self.has_been_initialized:
            tensors = list(branch_tensors.values())
            shape = tensors[0].shape
            assert all(tensor.shape == shape for tensor in tensors)
            self._custom_init(shape[-1])

        assert self.has_been_initialized
        x = torch.stack(list(branch_tensors.values()), dim=-2)

        # attention
        attention = x
        for layer in self.attention_layers:
            attention = layer(attention)
            attention = self.act(attention)
            attention = self.attention_norm(attention)
        attention = self.attention_output_layer(attention)
        attention = self.softmax(attention)

        # normal transformation
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
            x = self.norm(x)
        x = self.output_layer(x)

        out = torch.sum(attention * x, dim=-2)

        if self.residual:
            out = out + branch_tensors[this_index]

        return out

    @staticmethod
    def get_name() -> str:
        return "simple_attention"


BLOCK_DICT = {
    Class.get_name(): Class
    for Class in (
        SumFusion,
        MeanFusion,
        MaxFusion,
        SimpleConvFusion,
        SimpleAttentionFusion,
    )
}
