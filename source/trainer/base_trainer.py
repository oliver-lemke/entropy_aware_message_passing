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
from utils.config import Config
from utils.eval_metrics import metrics
from utils.logs import Logger, add_prefix_to_dict, combine_dicts

config = Config()
logger = Logger()


class BaseTrainer:
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

    def seed(self):
        """
        Set seeds for reproducibility.
        """
        seed = config["seed"]

        # Python's built-in random module
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

        # PyTorch Geometric (if necessary)
        # PyG does not have a specific seed setting, but it relies on PyTorch

        # Ensuring that PyTorch behaves deterministically (may impact performance)
        # Uncomment if deterministic behavior is required
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        # Setting seed for Python's 'os' module for file-system related operations
        os.environ["PYTHONHASHSEED"] = str(seed)

    def prepare_loaders(self):
        logger.info("Preparing dataloaders")
        transform = DatasetFactory.get_transform(self.transform_config)
        self.dataset = DatasetFactory().get_dataset(transform=transform)
        self.input_dim, self.output_dim = self.dataset.get_input_output_dim()

        self.loader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=config["hyperparameters"]["train"]["batch_size"],
        )

    def build_model(self):
        logger.info("Building model")
        learning_rate = config["hyperparameters"]["train"]["learning_rate"]
        weight_decay = config["hyperparameters"]["train"]["weight_decay"]
        epochs = config["hyperparameters"]["train"]["epochs"]

        self.model, self.transform_config = ModelFactory().get_model(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.model.to(config["device"])

        self.loss = torch.nn.CrossEntropyLoss(reduction="sum")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=0
        )

    def step(self, data):
        self._step += 1
        # training step
        self.model.train()
        data = data.to(config["device"])

        # gradient descent
        self.optimizer.zero_grad()
        pred, intermediate_representations = self.model(data)
        loss = self.loss(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.model.clamp_learnables()
        self._log(data, pred, loss, intermediate_representations)

    def one_epoch(self):
        logger.info(
            f"Training epoch {self._epoch} / {config['hyperparameters']['train']['epochs']}"
        )
        for data in self.loader:
            self.step(data)
        self.scheduler.step()

    def train(self):
        # so logging starts at 1
        if config["wandb"]["enable"]:
            wandb.log({})
        logger.info(str(config))

        logger.info("Saving config file")
        config_save_path = os.path.join(self.output_dir, "config.json")
        with open(config_save_path, "w", encoding="UTF-8") as file:
            json.dump(config.get_config(), file)

        if config["resume_training"]:
            path = config.get_subpath("resume_from")
            logger.info(f"Resuming training from {path}")
            self._load_state(path)

        for epoch in range(1, config["hyperparameters"]["train"]["epochs"] + 1):
            self._epoch = epoch
            self.one_epoch()

            path_ending = f"epoch_{epoch}"
            if epoch % config["hyperparameters"]["train"]["save_every"] == 0:
                path = os.path.join(self.checkpoints_dir, path_ending)
                self._save_state(path)
            if self.best_checkpoint:
                path = os.path.join(self.best_checkpoints_dir, path_ending)
                self._clear_best_checkpoint_dir()
                self._save_state(path)
                self.best_checkpoint = False
        self._close()

    def _make_output_dir(self):
        experiments_folder = config.get_subpath("output")
        self.id = name_stem = config.get_name_stem()
        self.output_dir = os.path.join(experiments_folder, name_stem)

        os.makedirs(experiments_folder, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=False)

        self.checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        self.best_checkpoints_dir = os.path.join(self.output_dir, "best_checkpoints")
        self.tensorboard_dir = os.path.join(self.output_dir, "tensorboard")
        self.wandb_dir = os.path.join(self.output_dir, "wandb")

        os.makedirs(self.checkpoints_dir, exist_ok=False)
        os.makedirs(self.best_checkpoints_dir, exist_ok=False)
        os.makedirs(self.tensorboard_dir, exist_ok=False)
        os.makedirs(self.wandb_dir, exist_ok=False)

    def _clear_best_checkpoint_dir(self):
        directory_path = self.best_checkpoints_dir
        if os.path.exists(directory_path):
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    def _save_state(self, path):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, config["save_names"]["weights"])
        optimizer_path = os.path.join(path, config["save_names"]["optimizer"])

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

    def _load_state(self, path):
        model_path = os.path.join(path, config["save_names"]["weights"])
        optimizer_path = os.path.join(path, config["save_names"]["optimizer"])

        model_state_dict = torch.load(model_path, map_location=config["device"])
        optimizer_state_dict = torch.load(optimizer_path, map_location=config["device"])

        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)

    def _log_all(self, scalar_metrics: dict, other_wandb: dict):
        for name, value in scalar_metrics.items():
            self.writer.add_scalar(name, value, self._step)
            logger.info(f"{name}: {value:.5f}")
        if config["wandb"]["enable"]:
            log_dict = combine_dicts(**scalar_metrics, **other_wandb)
            wandb.log(log_dict)

    def _log(self, data, pred, loss, int_reps):
        # ENtropy Object
        A = torch_geometric.utils.to_dense_adj(data.edge_index).squeeze()
        entropy = Entropy(T=1.0, A=A)
        # metrics
        self.model.eval()
        with torch.no_grad():
            # normal metrics
            train_metrics = metrics(
                pred[data.train_mask], data.y[data.train_mask], reduction="mean"
            )
            train_metrics["total_loss"] = loss.item()
            val_metrics = metrics(
                pred[data.val_mask], data.y[data.val_mask], reduction="mean"
            )
            # validation loss
            val_loss = self.loss(pred[data.val_mask], data.y[data.val_mask])
            val_metrics["total_loss"] = val_loss.item()
            if val_loss < self.prev_val_loss:
                self.prev_val_loss = val_loss
                self.best_checkpoint = True

            total_metrics = {}
            other = {}
            energies, entropies = [], []

            # calculate energy and entropy from intermediate representations
            for layer, int_rep in int_reps.items():
                energy_metric = entropy.dirichlet_energy(int_rep).mean()
                entropy_metric = entropy.entropy(int_rep)
                energies.append(energy_metric)
                entropies.append(entropy_metric)
                if isinstance(layer, int):
                    total_metrics[f"energy_over_epoch/layer{layer:04d}"] = energy_metric
                    total_metrics[
                        f"entropy_over_epoch/layer{layer:04d}"
                    ] = entropy_metric
                else:
                    total_metrics[f"energy_over_epoch/layer{layer}"] = energy_metric
                    total_metrics[f"entropy_over_epoch/layer{layer}"] = entropy_metric

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

            # prepare for logging
            train_metrics = add_prefix_to_dict(train_metrics, "train/")
            val_metrics = add_prefix_to_dict(val_metrics, "val/")
            total_metrics = add_prefix_to_dict(total_metrics, "total/")
            scalar_metrics = combine_dicts(
                **train_metrics, **val_metrics, **total_metrics
            )
            self._log_all(scalar_metrics=scalar_metrics, other_wandb=other)

    def _close(self):
        wandb.finish(exit_code=0, quiet=False)


def flag_sanity_check(flags):
    """
    Checks the flags for compatibility
    """
    id(flags)
