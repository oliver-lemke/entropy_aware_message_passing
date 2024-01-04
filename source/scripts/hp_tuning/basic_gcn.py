import itertools

from trainer.base_trainer import BaseTrainer
from utils.config import Config


def new_config() -> Config:
    config = Config()
    config["model_type"] = "basic_gcn"
    config["wandb"]["enable"] = True
    config["wandb"]["extended"] = False
    config["wandb"]["project"] = "basic_gcn"
    return config


def main():
    model_type = "basic_gcn"
    repetitions = 2
    model_depths = (2, 3, 4, 8, 16, 32, 64)

    combinations = itertools.product(
        list(range(repetitions)),
        model_depths,
    )

    for (
        repetition,
        model_depth,
    ) in combinations:
        config = new_config()
        config["note"] = f"hp-d_{model_depth}-r_{repetition}"

        # model
        config["model_type"] = model_type
        config["model_parameters"][model_type]["depth"] = model_depth

        agent = BaseTrainer()
        agent.train()
        del agent


if __name__ == "__main__":
    main()
