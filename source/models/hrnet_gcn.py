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

        # get the different block types
        block_factory = BlockFactory()
        # ConvBlock is the type of block responsible for doing graph convolutions
        ConvBlock = block_factory.get_conv_block()
        # TransformBlocks do transformations of individual nodes within a branch
        TransformBlock = block_factory.get_transform_block()
        # FusionBlocks are responsible for fusing different branches
        FusionBlock = block_factory.get_fusion_block()
        # OutputBlock is only responsible for the final output of the model
        OutputBlock = block_factory.get_output_block()

        # this is the bottom branch of the network, which simply does convolutions
        self.base_network = nn.ModuleList(
            [ConvBlock(input_dim, hidden_dim)]
            + [ConvBlock(hidden_dim, hidden_dim) for _ in range(self.depth - 2)]
        )
        self.transform_blocks = self._setup_transform_blocks(TransformBlock)
        self.fusion_blocks = self._setup_fusion_blocks(FusionBlock)
        self.output_block = OutputBlock()
        self.down_projection = nn.Linear(hidden_dim, output_dim)

        if 0 in self.split_indices:
            middle_dim = (input_dim + hidden_dim) // 2
            self.initial_projection_layer = nn.Sequential(
                nn.Linear(input_dim, middle_dim),
                nn.ReLU(),
                nn.LayerNorm(middle_dim),
                nn.Linear(middle_dim, hidden_dim),
            )

        logger.debug(str(self))

    def _setup_transform_blocks(self, TransformBlock) -> nn.ModuleDict:
        transform_blocks = nn.ModuleDict()
        for branch_index, split_index in enumerate(self.split_indices):
            # the split_index is the point where the new branch splits off
            # the branch_index is the iudex of the branch (0 for top, etc.)
            # for each branch we create its own dict
            branch = nn.ModuleDict()
            for depth in range(split_index, self.depth):
                # each branch does one transformation at every level
                branch[str(depth)] = TransformBlock(self.hidden_dim, self.hidden_dim)
            transform_blocks[str(branch_index)] = branch
        return transform_blocks

    def _setup_fusion_blocks(self, FusionBlock) -> nn.ModuleDict:
        fusion_blocks = nn.ModuleDict()
        # set up fusion for all the extra branches
        for branch_index in range(self.bot_branch_idx):
            # the split_index is the point where the new branch splits off
            # the branch_index is the iudex of the branch (0 for top, etc.)
            # for each branch we create its own dict
            branch = nn.ModuleDict()
            for depth in self.split_indices[branch_index + 1 :]:
                # each branch has a fusion block when a new branch splits off
                # but off course not at its own
                # so the first branch to split off won't have a fusion block when
                # itself splits off, just at every step after
                branch[str(depth)] = FusionBlock()
            fusion_blocks[str(branch_index)] = branch

        # set up fusion for base branch
        # doesn't really fit into the loop, so we do it extra
        base_branch = nn.ModuleDict()
        for depth in self.split_indices[1:]:
            base_branch[str(depth)] = FusionBlock()
        fusion_blocks[str(self.bot_branch_idx)] = base_branch
        return fusion_blocks

    def _normal_step(
        self, tensors: dict[int, torch.Tensor], edge_index, current_depth: int
    ) -> dict[int, torch.Tensor]:
        for branch_idx, tensor in tensors.items():
            if branch_idx == self.bot_branch_idx:  # graph convolution
                network = self.base_network[current_depth]
                tensors[branch_idx] = network(tensor, edge_index)
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
        # the new branch has index "current_nr_branches - 1".
        # that is because the bottom branch is always the last index, meaning we have
        # "current_nr_branches - 1" branches before the new one
        tensors[current_nr_branches - 1] = new_branch_tensor
        return tensors

    def _fuse(
        self, tensors: dict[int, torch.Tensor], current_depth: int
    ) -> dict[int, torch.Tensor]:
        # temporary dict to store intermediate tensor computations
        # (so we don't use first fused tensors for later ones)
        new_tensors = {}
        for branch_idx in tensors:
            # get the fusion block responsible for that specific fusion
            network = self.fusion_blocks[str(branch_idx)][str(current_depth)]
            # give it all that fusion block
            new_tensors[branch_idx] = network(tensors, branch_idx)
        return new_tensors

    def forward(self, data):
        tensors = {self.bot_branch_idx: data.x}
        int_reps = {}  # {0: data.x}

        for current_depth in range(self.depth - 1):
            if current_depth in self.split_indices[1:]:
                tensors = self._fuse(tensors, current_depth)
            if current_depth in self.split_indices:
                tensors = self._branch(tensors, current_depth)
            tensors = self._normal_step(tensors, data.edge_index, current_depth)
            int_reps[current_depth + 1] = tensors[self.bot_branch_idx]

        out = self.output_block(tensors)
        int_reps["merged"] = out
        out = self.down_projection(out)
        int_reps["final"] = out
        return out, int_reps
