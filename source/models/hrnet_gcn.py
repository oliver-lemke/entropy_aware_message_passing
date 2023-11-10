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
        hidden_dim = params["hidden_dim"]
        self.depth = params["depth"]
        self.branch_indices = params["branches"]

        # block args
        block_factory = BlockFactory()
        ConvBlock = block_factory.get_conv_block()
        TransformBlock = block_factory.get_transform_block()
        FusionBlock = block_factory.get_fusion_block()

        self.base_network = nn.ModuleList(
            [ConvBlock(input_dim, hidden_dim)]
            + [ConvBlock(hidden_dim, hidden_dim) for _ in range(self.depth - 2)]
            + [ConvBlock(hidden_dim, output_dim)]
        )
        self.branch_networks = nn.ModuleDict()
        self.branch_tensors = {}

        self.current_nr_branches = 1
        self.bot_branch_idx = len(self.branch_indices)
        self.edge_index = None
        logger.debug(str(self))

    def _normal_step(self) -> None:
        for idx, tensor in self.branch_tensors.items():
            self.branch_tensors[idx] = self.branch_networks[idx](tensor)

    def _branch(self):
        # use the current tensor from the lowest branch
        bot_tensor = self.branch_tensors[self.bot_branch_idx]
        # put it into the next branching pathway
        self.branch_tensors[self.current_nr_branches] = bot_tensor
        # increment current #branches
        self.current_nr_branches += 1

    def _fuse(self) -> None:
        pass

    def forward(self, data):
        self.branches[self.bot_branch_idx] = data.x
        self.edge_index = data.edge_index

        for idx, conv in enumerate(self.base_network):
            if idx in self.branch_indices:
                self._branching_step()
            else:
                self._normal_step()

        return self.branch_tensors[self.bot_branch_idx]
