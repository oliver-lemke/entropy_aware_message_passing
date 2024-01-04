import itertools

from trainer.base_trainer import BaseTrainer
from utils.config import Config


def new_config() -> Config:
    config = Config()
    config["wandb"]["enable"] = True
    config["wandb"]["extended"] = False
    config["dataset"] = "planetoid"
    config["dataset_parameters"]["planetoid"]["name"] = "Cora"
    config["wandb"]["project"] = "Cora"
    config["hyperparameters"]["train"]["epochs"] = 100
    return config


def main():
    model_types = ("basic_gcn", "entropic_gcn", "pairnorm_gcn", "g2")
    model_depths = (2, 4, 8, 16, 32, 64)

    combinations = itertools.product(
        model_types,
        model_depths,
    )

    for (
        model_type,
        model_depth,
    ) in combinations:
        config = new_config()
        config["note"] = f"{model_type}-d_{model_depth}"

        # model
        config["model_type"] = model_type
        config["model_parameters"][model_type]["depth"] = model_depth

        agent = BaseTrainer()
        agent.train()
        del agent


if __name__ == "__main__":
    main()
