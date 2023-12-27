"""
File is used for testing the actual model.
"""
import os

import torch

import wandb
from models import ModelFactory
from physics.physics import Entropy
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_dense_adj
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


def new_config() -> Config:
    config = Config()
    config["wandb"]["enable"] = True
    config["wandb"]["project"] = "energy-per-layer-testing"
    return config


def grid_4_neighbors(rows, cols):
    edges = []

    for i in range(rows * cols):
        # Compute the row and column indices of the node
        r = i // cols
        c = i % cols

        # Add an edge to the left neighbor if it exists
        if c > 0:
            edges.append([i, i - 1])

        # Add an edge to the right neighbor if it exists
        if c < cols - 1:
            edges.append([i, i + 1])

        # Add an edge to the top neighbor if it exists
        if r > 0:
            edges.append([i, i - cols])

        # Add an edge to the bottom neighbor if it exists
        if r < rows - 1:
            edges.append([i, i + cols])

    edge_index = torch.tensor(edges, dtype=torch.long)

    # Transpose to match COO format
    return edge_index.t()


class BaseTester:
    def __init__(self):
        self.model = None
        self.transform_config = None
        self.dataset = None
        self.input_dim = -1
        self.output_dim = -1

        self._make_output_dir()
        if config["wandb"]["enable"]:
            self.run = wandb.init(
                entity=config["wandb"]["entity"],
                project=config["wandb"]["project"],
                group=config["wandb"]["group"],
                config=config.get_config(),
                dir=self.wandb_dir,
                id=self.id,
            )

        # load hyperparameters
        self.params = config["model_parameters"]["entropic_gcn"]

        self.T = self.params["temperature"]["value"]
        self.norm_energy = self.params["normalize_energies"]
        self.norm_dist = self.params["normalize_distribution"]

    def _make_output_dir(self):
        experiments_folder = config.get_subpath("output")
        self.id = name_stem = config.get_name_stem()
        self.output_dir = os.path.join(experiments_folder, name_stem)

        os.makedirs(experiments_folder, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=False)

        self.wandb_dir = os.path.join(self.output_dir, "wandb")

        os.makedirs(self.wandb_dir, exist_ok=False)

    def prepare_dataset(self):
        logger.info("Preparing dataset")
        edge_index, _ = add_self_loops(grid_4_neighbors(10, 10))
        x = torch.rand(100, 1)

        self.dataset = Data(x=x, edge_index=edge_index).to(config["device"])

        self.input_dim, self.output_dim = 1, 64

        # initialize Entropy class
        A = to_dense_adj(self.dataset.edge_index).squeeze()

        # TODO: load these bools from config
        self.entropy = Entropy(
            A=A, norm_energy=self.norm_energy, norm_dist=self.norm_dist
        )

    def prepare_model(self):
        self.model, _ = ModelFactory().get_model(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.model.to(config["device"])

    def calculate_energy(self):
        return self.entropy.total_dirichlet_energy(self.model(self.dataset)[0])

    def calculate_entropy(self):
        return self.entropy.entropy(self.model(self.dataset)[0], self.T)

    def test_energy_per_layer(self):
        log_dict = {}
        # for model_type in ["basic_gcn"]:
        for model_type in ["basic_gcn", "hrnet_gcn", "entropic_gcn", "g2"]:
            data_energy = []
            data_entropy = []
            config["model_type"] = model_type
            wandb.define_metric("depth")
            wandb.define_metric("energy/*", step_metric="depth")
            wandb.define_metric("entropy/*", step_metric="depth")
            for depth in range(10, 1000, 100):
                config["model_parameters"][model_type]["depth"] = depth
                self.prepare_dataset()
                self.prepare_model()
                log_dict = {
                    "depth": depth,
                    f"energy/{model_type}": self.calculate_energy(),
                    f"entropy/{model_type}": self.calculate_entropy(),
                }
                wandb.log(log_dict)
