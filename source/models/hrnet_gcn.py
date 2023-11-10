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

        self.current_nr_branches = 1
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
        self.branch_tensors = {}
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
        for branch_index in range(1, self.bot_branch_idx):
            branch = nn.ModuleDict()
            for depth in self.split_indices[branch_index:]:
                branch[str(depth)] = FusionBlock()
            fusion_blocks[str(branch_index)] = branch
        return fusion_blocks

    def _normal_step(self, current_depth: int) -> None:
        for branch_idx, tensor in self.branch_tensors.items():
            network = self.branch_networks[branch_idx][current_depth]
            self.branch_tensors[branch_idx] = network(tensor)

    def _branch(self):
        # use the current tensor from the lowest branch
        bot_tensor = self.branch_tensors[self.bot_branch_idx]
        # put it into the next branching pathway
        self.branch_tensors[self.current_nr_branches] = bot_tensor
        # increment current #branches
        self.current_nr_branches += 1

    def _fuse(self, current_depth: int) -> None:
        pass

    def forward(self, data):
        self.branches[self.bot_branch_idx] = data.x
        self.edge_index = data.edge_index

        for current_depth, conv in range(self.depth):
            if current_depth in self.split_indices:
                self._branching_step()
            else:
                self._normal_step()

        return self.branch_tensors[self.bot_branch_idx]
