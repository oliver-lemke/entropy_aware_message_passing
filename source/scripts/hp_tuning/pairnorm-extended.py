import itertools

from trainer.base_trainer import BaseTrainer
from utils.config import Config


def new_config() -> Config:
    config = Config()
    config["model_type"] = "pairnorm_gcn"
    config["wandb"]["enable"] = True
    config["wandb"]["extended"] = True
    config["wandb"]["project"] = "pairnorm-extended"
    return config


def main():
    model_type = "pairnorm_gcn"
    repetitions = 1
    model_depths = (16,)
    modes = ("PN", "PN-SI", "PN-SCS", None)

    combinations = itertools.product(list(range(repetitions)), model_depths, modes)

    for repetition, model_depth, mode in combinations:
        config = new_config()
        config["note"] = f"hp-d_{model_depth}-r_{repetition}-m_{mode}"

        # model
        config["model_type"] = model_type
        config["model_parameters"][model_type]["depth"] = model_depth
        config["model_parameters"][model_type]["norm_mode"] = mode

        agent = BaseTrainer()
        agent.train()
        del agent


if __name__ == "__main__":
    main()
