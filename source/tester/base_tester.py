"""
File is used for testing the actual model.
"""
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import wandb
from datasets import DatasetFactory
from models import ModelFactory
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, to_dense_adj, to_networkx
from utils.config import Config
from utils.eval_metrics import metrics
from utils.logs import Logger

config = Config()
logger = Logger()


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


def dirichlet_energy(x, edge_index):
    # x is a N by F tensor, where N is the number of nodes and F is the number of features
    # edge_index is a 2 by E tensor, where E is the number of edges
    # Convert the edge_index to a dense adjacency matrix
    adj = to_dense_adj(edge_index)[0]

    # Compute the degree matrix
    deg = torch.diag(torch.sum(adj, dim=1))

    # Compute the Laplacian matrix
    lap = deg - adj
    print(lap.shape)
    print(x)
    print(torch.mm(torch.mm(x.t(), lap), x).shape)
    # Compute the Dirichlet energy
    energy = torch.trace(torch.mm(torch.mm(x.t(), lap), x))

    # Return the energy
    return energy


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
        
        self.prepare_dataset()
        self.prepare_model()

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

        self.dataset = Data(x=x, edge_index=edge_index)

        self.input_dim, self.output_dim = 1, 64

    def prepare_model(self):
        self.model, _ = ModelFactory().get_model(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.model.to(config["device"])

    def test(self):
        return dirichlet_energy(self.model(self.dataset)[0], self.dataset.edge_index)
