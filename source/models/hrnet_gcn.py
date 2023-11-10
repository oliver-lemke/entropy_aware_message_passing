import torch
from torch import nn

from models.model_utils import BlockFactory
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class HRNetGCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        params = config["model_parameters"]["hrnet_gcn"]
        self.hidden_dim = hidden_dim = params["hidden_dim"]
        self.depth = params["depth"]
        self.split_indices = params["branches"]

        self.bot_branch_idx = len(self.split_indices)
        self.edge_index = None

        # get the different block types
        block_factory = BlockFactory()
        ConvBlock = block_factory.get_conv_block()
        TransformBlock = block_factory.get_transform_block()
        FusionBlock = block_factory.get_fusion_block()

        self.base_network = nn.ModuleList(
            [ConvBlock(input_dim, hidden_dim)]
            + [ConvBlock(hidden_dim, hidden_dim) for _ in range(self.depth - 2)]
            + [ConvBlock(hidden_dim, output_dim)]
        )
        self.transform_blocks = self._setup_transform_blocks(TransformBlock)
        self.fusion_blocks = self._setup_fusion_blocks(FusionBlock)

        if 0 in self.split_indices:
            middle_dim = (input_dim + hidden_dim) // 2
            self.initial_projection_layer = nn.Sequential(
                nn.Linear(input_dim, middle_dim),
                nn.ReLU(),
                nn.Linear(middle_dim, hidden_dim),
            )

        logger.debug(str(self))

    def _setup_transform_blocks(self, TransformBlock) -> nn.ModuleDict:
        transform_blocks = nn.ModuleDict()
        for branch_index, split_index in enumerate(self.split_indices):
            branch = nn.ModuleDict()
            for depth in range(split_index, self.depth):
                branch[str(depth)] = TransformBlock(self.hidden_dim, self.hidden_dim)
            transform_blocks[str(branch_index)] = branch
        return transform_blocks

    def _setup_fusion_blocks(self, FusionBlock) -> nn.ModuleDict:
        fusion_blocks = nn.ModuleDict()
        # set up fusion for all the extra branches
        for branch_index in range(self.bot_branch_idx):
            branch = nn.ModuleDict()
            for depth in self.split_indices[branch_index + 1 :]:
                branch[str(depth)] = FusionBlock()
            fusion_blocks[str(branch_index)] = branch

        # set up fusion for base branch
        base_branch = nn.ModuleDict()
        for depth in self.split_indices[1:]:
            base_branch[str(depth)] = FusionBlock()
        fusion_blocks[str(self.bot_branch_idx)] = base_branch
        return fusion_blocks

    def _normal_step(
        self, tensors: dict[int, torch.Tensor], current_depth: int
    ) -> dict[int, torch.Tensor]:
        for branch_idx, tensor in tensors.items():
            if branch_idx == self.bot_branch_idx:  # graph convolution
                network = self.base_network[current_depth]
                tensors[branch_idx] = network(tensor, self.edge_index)
            else:  # transform
                network = self.transform_blocks[str(branch_idx)][str(current_depth)]
                tensors[branch_idx] = network(tensor)
        return tensors

    def _branch(
        self, tensors: dict[int, torch.Tensor], current_depth: int
    ) -> dict[int, torch.Tensor]:
        # use the current tensor from the lowest branch
        new_branch_tensor = tensors[self.bot_branch_idx]
        # if we branch before the first convolution, project down
        if current_depth == 0:
            new_branch_tensor = self.initial_projection_layer(new_branch_tensor)
        # put it into the next branching pathway
        current_nr_branches = len(tensors)
        tensors[current_nr_branches - 1] = new_branch_tensor
        # increment current #branches
        return tensors

    def _fuse(
        self, tensors: dict[int, torch.Tensor], current_depth: int
    ) -> dict[int, torch.Tensor]:
        for branch_idx in tensors:
            network = self.fusion_blocks[str(branch_idx)][str(current_depth)]
            tensors[branch_idx] = network(tensors, branch_idx)
        return tensors

    def forward(self, data):
        tensors = {self.bot_branch_idx: data.x}
        self.edge_index = data.edge_index

        for current_depth in range(self.depth):
            if current_depth in self.split_indices[1:]:
                tensors = self._fuse(tensors, current_depth)
            if current_depth in self.split_indices:
                tensors = self._branch(tensors, current_depth)
            tensors = self._normal_step(tensors, current_depth)

        return tensors[self.bot_branch_idx]
