"""
File is used for training the actual model.
"""
import json
import os
import random
import shutil

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import torch_geometric
import wandb
from datasets import DatasetFactory
from models import ModelFactory
from physics.physics import Entropy
from torch_geometric.loader import DataLoader
from trainer.base_trainer import BaseTrainer
from utils.config import Config
from utils.eval_metrics import metrics
from utils.logs import Logger, add_prefix_to_dict, combine_dicts

config = Config()
logger = Logger()


class MultiGraphTrainer(BaseTrainer):
    def __init__(self):
        self.scheduler = None
        self.optimizer = None
        self.loss = None
        self.transform_config = None
        self.model = None
        self.dataset = None
        self.loader = None
        self.input_dim = -1
        self.output_dim = -1
        self.prev_val_loss = np.infty
        self.best_checkpoint = False
        self._step = 0
        self._epoch = 0

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

        self.seed()
        self.prepare_loaders()
        self.build_model()
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

    def prepare_loaders(self):
        logger.info("Preparing dataloaders")
        transform = DatasetFactory.get_transform(self.transform_config)
        self.dataset = DatasetFactory().get_dataset(transform=transform)
        self.input_dim, self.output_dim = self.dataset.get_input_output_dim()

        self.train_loader = DataLoader(
            self.dataset.train,
            shuffle=True,
            batch_size=config["hyperparameters"]["train"]["batch_size"],
        )

        self.test_loader = DataLoader(
            self.dataset.test,
            shuffle=True,
            batch_size=config["hyperparameters"]["train"]["batch_size"],
        )

        self.val_loader = DataLoader(
            self.dataset.val,
            shuffle=True,
            batch_size=config["hyperparameters"]["train"]["batch_size"],
        )

    def step(self, data):
        self._step += 1
        # training step
        self.model.train()
        data = data.to(config["device"])

        # gradient descent
        self.optimizer.zero_grad()
        pred, intermediate_representations = self.model(data)
        loss = self.loss(pred, data.y)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.model.clamp_learnables()
        self._log(data, pred, loss, intermediate_representations)

    def one_epoch(self):
        logger.info(
            f"Training epoch {self._epoch} / {config['hyperparameters']['train']['epochs']}"
        )
        for data in self.train_loader:
            self.step(data)
        self.scheduler.step()

    def _log(self, data, pred, loss, int_reps):
        # ENtropy Object
        A = torch_geometric.utils.to_dense_adj(data.edge_index).squeeze()
        entropy = Entropy(A=A)
        # metrics
        self.model.eval()
        with torch.no_grad():
            # normal metrics
            train_metrics = metrics(pred, data.y, reduction="mean")
            train_metrics["total_loss"] = loss.item()
            # val_metrics = metrics(
            #     pred[data.val_mask], data.y[data.val_mask], reduction="mean"
            # )
            # # validation loss
            # val_loss = self.loss(pred[data.val_mask], data.y[data.val_mask])
            # val_metrics["total_loss"] = val_loss.item()
            # if val_loss < self.prev_val_loss:
            #     self.prev_val_loss = val_loss
            #     self.best_checkpoint = True
