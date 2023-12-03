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
from utils.config import Config
from utils.eval_metrics import metrics
from utils.logs import Logger

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

    def compute_metrics_dict(self, data, pred, int_reps, entropy):
        # metrics
        self.model.eval()
        with torch.no_grad():
            train_metrics = metrics(pred[data.train_mask], data.y[data.train_mask])
            val_metrics = metrics(pred[data.val_mask], data.y[data.val_mask])
            total_metrics = {}
            for layer, int_rep in int_reps.items():
                energy_metric = entropy.dirichlet_energy(int_rep).mean()
                entropy_metric = entropy.entropy(int_rep)
                total_metrics[f"total/energy_mean/layer{layer:04d}"] = energy_metric
                total_metrics[f"total/entropy/layer{layer:04d}"] = entropy_metric

            val_loss = self.loss(pred[data.val_mask], data.y[data.val_mask])
            val_metrics["total_loss"] = val_loss.item()
            if val_loss < self.prev_val_loss:
                self.prev_val_loss = val_loss
                self.best_checkpoint = True

        return train_metrics, val_metrics, total_metrics

    def step(self, data):
        # training step
        self.model.train()
        data = data.to(config["device"])

        A = torch_geometric.utils.to_dense_adj(data.edge_index).squeeze()
        entropy = Entropy(T=1.0, A=A)

        # gradient descent
        self.optimizer.zero_grad()
        pred, intermediate_representations = self.model(data)
        loss = self.loss(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()

        train_metrics, val_metrics, total_metrics = self.compute_metrics_dict(
            data, pred, intermediate_representations, entropy
        )

        # make ready for return
        train_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
        val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
        full_metrics = {
            "train/total_loss": loss.item(),
            **train_metrics,
            **val_metrics,
            **total_metrics,
        }

        return (
            full_metrics,
            torch.sum(data.train_mask).item(),
            torch.sum(data.val_mask).item(),
        )

    def one_epoch(self, epoch):
        logger.info(
            f"Training epoch {epoch} / {config['hyperparameters']['train']['epochs']}"
        )

        total_metrics = defaultdict(float)
        total_train, total_val = 0, 0

        for data in self.loader:
            step_metrics, nr_train, nr_val = self.step(data)

            for k in step_metrics.keys():
                total_metrics[k] += step_metrics[k]
            total_train += nr_train
            total_val += nr_val

        self.scheduler.step()

        for k, v in total_metrics.items():
            if k.startswith("train/"):
                total_metrics[k] = v / total_train
            elif k.startswith("val/"):
                total_metrics[k] = v / total_val
        return total_metrics

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
            epoch_metrics = self.one_epoch(epoch)

            path_ending = f"epoch_{epoch}"
            if epoch % config["hyperparameters"]["train"]["save_every"] == 0:
                path = os.path.join(self.checkpoints_dir, path_ending)
                self._save_state(path)
            if self.best_checkpoint:
                path = os.path.join(self.best_checkpoints_dir, path_ending)
                self._clear_best_checkpoint_dir()
                self._save_state(path)
                self.best_checkpoint = False

            self._log(epoch, **epoch_metrics)
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

    def _log(self, epoch: int, **log_dict):
        for name, value in log_dict.items():
            self.writer.add_scalar(name, value, epoch)
            logger.info(f"{name}: {value:.5f}")
        if config["wandb"]["enable"]:
            wandb.log(log_dict)

    def _close(self):
        wandb.finish(exit_code=0, quiet=False)


def flag_sanity_check(flags):
    """
    Checks the flags for compatibility
    """
    id(flags)
