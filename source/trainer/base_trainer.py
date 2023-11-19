"""
File is used for training the actual model.
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import wandb
from datasets import DatasetFactory
from models import ModelFactory
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

        self.prepare_loaders()
        self.build_model()
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

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
        # training step
        self.model.train()
        data = data.to(config["device"])

        # gradient descent
        self.optimizer.zero_grad()
        pred = self.model(data)
        loss = self.loss(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()

        # metrics
        self.model.eval()
        with torch.no_grad():
            train_metrics = metrics(pred[data.train_mask], data.y[data.train_mask])
            val_metrics = metrics(pred[data.val_mask], data.y[data.val_mask])

        # make ready for return
        train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        full_metrics = {"loss": loss.item(), **train_metrics, **val_metrics}

        return full_metrics, len(data.train_mask), len(data.val_mask)

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

        total_metrics["loss"] = total_metrics["loss"] / total_train
        for k, v in total_metrics.items():
            if k.startswith("train_"):
                total_metrics[k] = v / total_train
            elif k.startswith("val_"):
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

            if epoch % config["hyperparameters"]["train"]["save_every"] == 0:
                path = os.path.join(self.checkpoints_dir, f"epoch_{epoch}")
                self._save_state(path)

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
