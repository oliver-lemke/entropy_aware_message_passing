"""
File is used for training the actual model.
"""
import json
import os
import random
import shutil
from collections import defaultdict

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
        self.val_metrics = defaultdict(int)

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
            shuffle=False,
            batch_size=config["hyperparameters"]["train"]["batch_size"],
        )

        self.val_loader = DataLoader(
            self.dataset.val,
            shuffle=False,
            batch_size=config["hyperparameters"]["train"]["batch_size"],
        )

    def train_step(self, data):
        self._step += 1
        # training step
        self.model.train()
        data = data.to(config["device"])

        # gradient descent
        self.optimizer.zero_grad()
        pred, intermediate_representations, log_data = self.model(data)
        loss = self.loss(pred, data.y)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.model.clamp_learnables()
        self._log_train(data, pred, loss, intermediate_representations, log_data)

    def val_step(self, data):
        # training step
        self.model.train()
        data = data.to(config["device"])

        # gradient descent
        with torch.no_grad():
            self.optimizer.zero_grad()
            pred, _, _ = self.model(data)
            loss = self.loss(pred, data.y)
            self.model.clamp_learnables()

        self._store_val(data, pred, loss)

    def one_epoch(self):
        logger.info(
            f"Training epoch {self._epoch} / {config['hyperparameters']['train']['epochs']}"
        )
        for data in self.train_loader:
            self.train_step(data)
        n = 0
        for data in self.val_loader:
            self.val_step(data)
            n += 1
        self._log_val(n)
        self.scheduler.step()

    def _log_train(self, data, pred, loss, int_reps, log_data):
        # ENtropy Object
        A = torch_geometric.utils.to_dense_adj(data.edge_index).squeeze()
        entropy = Entropy(A=A)
        # metrics
        self.model.eval()
        with torch.no_grad():
            # normal metrics
            train_metrics = metrics(pred, data.y, reduction="mean")
            train_metrics["total_loss"] = loss.item()
        # prepare for logging
        scalar_metrics = add_prefix_to_dict(train_metrics, "train/")
        scalar_metrics = combine_dicts(**scalar_metrics, **log_data)

        other = {}
        if config["wandb"]["extended"]:
            with torch.no_grad():
                total_metrics = {}
                energies, entropies = [], []
                # calculate energy and entropy from intermediate representations
                for layer, int_rep in int_reps.items():
                    energy_metric = entropy.dirichlet_energy(int_rep).mean()
                    entropy_metric = entropy.entropy(int_rep, 1.0)
                    energies.append(energy_metric)
                    entropies.append(entropy_metric)
                    if isinstance(layer, int):
                        total_metrics[
                            f"energy_over_epoch/layer{layer:04d}"
                        ] = energy_metric
                        total_metrics[
                            f"entropy_over_epoch/layer{layer:04d}"
                        ] = entropy_metric
                    else:
                        total_metrics[f"energy_over_epoch/layer{layer}"] = energy_metric
                        total_metrics[
                            f"entropy_over_epoch/layer{layer}"
                        ] = entropy_metric

                # plot over layers
                table_data = [
                    [layer, energy.item(), entropy.item()]
                    for layer, (energy, entropy) in enumerate(zip(energies, entropies))
                ]
                # Create a wandb Table
                energy_entropy_table = wandb.Table(
                    data=table_data, columns=["Layer", "Energy", "Entropy"]
                )
                # Log the table as line plots
                other[f"energy_over_layers/step{self._step:04d}"] = wandb.plot.line(
                    table=energy_entropy_table,
                    x="Layer",
                    y="Energy",
                    title=f"energy_over_layers/step{self._step:04d}",
                )
                other[f"entropy_over_layers/step{self._step:04d}"] = wandb.plot.line(
                    table=energy_entropy_table,
                    x="Layer",
                    y="Entropy",
                    title=f"entropy_over_layers/step{self._step:04d}",
                )

                # histogram of last
                last_layer = max([k for k in int_reps if isinstance(k, int)])
                energy_metric = entropy.dirichlet_energy(int_reps[last_layer])
                other[f"histograms/final_energy_step{self._step}"] = wandb.Histogram(
                    energy_metric.cpu().numpy()
                )
                total_metrics = add_prefix_to_dict(total_metrics, "total/")
                scalar_metrics = combine_dicts(**scalar_metrics, **total_metrics)

        self._log_all(scalar_metrics=scalar_metrics, other_wandb=other)

    def _store_val(self, data, pred, loss):
        self.model.eval()
        with torch.no_grad():
            # normal metrics
            val_metrics = metrics(pred, data.y, reduction="mean")
            for key, value in val_metrics.items():
                self.val_metrics[key] += value
            # validation loss
            self.val_metrics["total_loss"] += loss.item()
            if loss < self.prev_val_loss:
                self.prev_val_loss = loss
                self.best_checkpoint = True

    def _log_val(self, nb_items):
        val_metrics = {}
        for key, value in self.val_metrics.items():
            val_metrics[key] = value / nb_items

        val_metrics = add_prefix_to_dict(val_metrics, "val/")
        logger.info(val_metrics)
        self._log_all(scalar_metrics=val_metrics, other_wandb={})
        self.val_metrics = defaultdict(int)
