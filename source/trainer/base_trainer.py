"""
File is used for training the actual model.
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb
from datasets import dataset_template
from models import ModelFactory
from tqdm import tqdm
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
        self.train_loader = None
        self.val_loader = None
        self.logger = Logger()
        self.prev_val_loss = np.infty

        self.build_model()
        self.prepare_loaders()
        self._make_output_dir()
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        wandb.init(
            project=config["project_name"],
            config=config.get_config(),
            dir=self.wandb_dir,
            id=self.id,
        )

    def prepare_loaders(self):
        self.logger.info("Preparing dataloaders")
        transforms = dataset_template.compute_transforms(self.transform_config)

        train_dataset = dataset_template.TemplateDataset(
            split="train",
            transformations=transforms,
            flags=None,
        )
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=config["hyperparameters"]["train"]["batch_size"],
        )

        val_dataset = dataset_template.TemplateDataset(
            split="val",
            transformations=transforms,
            flags=None,
        )
        self.val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=config["hyperparameters"]["val"]["batch_size"],
        )

    def build_model(self):
        self.logger.info("Building model")
        learning_rate = config["hyperparameters"]["train"]["learning_rate"]
        weight_decay = config["hyperparameters"]["train"]["weight_decay"]
        epochs = config["hyperparameters"]["train"]["epochs"]

        self.model, self.transform_config = ModelFactory().get_model(
            config["model_type"], in_channels=3
        )
        self.model.to(config["device"])

        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=0
        )

    def step(self, data):
        input_ = data["input"].to(config["device"])
        target = data["target"].to(config["device"])

        self.optimizer.zero_grad()

        pred = self.model(input_)
        loss = self.loss(pred, target)
        loss_metrics = metrics(pred, target)

        full_metrics = {"loss": loss.item(), **loss_metrics}

        return loss, full_metrics

    def train_one_epoch(self, epoch):
        self.logger.info(
            f"Training epoch {epoch} / {config['hyperparameters']['train']['epochs']}"
        )

        total_metrics = defaultdict(float)

        self.model.train()
        for data in tqdm(self.train_loader):
            loss, train_metrics = self.step(data)

            loss.backward()
            self.optimizer.step()

            for k in train_metrics.keys():
                total_metrics[k] += train_metrics[k]

        self.scheduler.step()
        total_metrics = {
            f"train_{k}": v / len(self.train_loader) for k, v in total_metrics.items()
        }
        return total_metrics

    def train(self):
        # so logging starts at 1
        wandb.log({})
        self.logger.info(str(config))

        self.logger.info("Saving config file")
        config_save_path = os.path.join(self.output_dir, "config.json")
        with open(config_save_path, "w", encoding="UTF-8") as file:
            json.dump(config.get_config(), file)

        if config["resume_training"]:
            path = config.get_subpath("resume_from")
            self.logger.info(f"Resuming training from {path}")
            self._load_state(path)

        for epoch in range(1, config["hyperparameters"]["train"]["epochs"] + 1):
            train_metrics = self.train_one_epoch(epoch)

            if epoch % config["hyperparameters"]["train"]["save_every"] == 0:
                path = os.path.join(self.checkpoints_dir, f"epoch_{epoch}")
                self._save_state(path)

            if epoch % config["hyperparameters"]["val"]["val_every"] == 0:
                val_metrics = self.validate(epoch)
                self._log(epoch, **train_metrics, **val_metrics)
            else:
                self._log(epoch, **train_metrics)

        self._close()

    def validate(self, epoch):
        self.logger.info(f"Validating: Epoch {epoch}")
        total_loss = 0
        total_metrics = defaultdict(float)

        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.train_loader):
                _, val_metrics = self.step(data)

                for k in val_metrics.keys():
                    total_metrics[k] += val_metrics[k]

        if total_loss / len(self.val_loader) < self.prev_val_loss:
            self.prev_val_loss = total_loss / len(self.val_loader)
            path = os.path.join(self.best_checkpoints_dir, f"epoch_{epoch}")
            self._save_state(path)

        total_metrics = {
            f"val_{k}": v / len(self.val_loader) for k, v in total_metrics.items()
        }
        return total_metrics

    def _make_output_dir(self):
        self.logger.info("Creating experiment log directories")
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
            self.logger.info(f"{name}: {value:.5f}")
        wandb.log(log_dict)

    def _close(self):
        wandb.finish(exit_code=0, quiet=False)


def flag_sanity_check(flags):
    """
    Checks the flags for compatibility
    """
    id(flags)
