import itertools

from trainer.base_trainer import BaseTrainer
from utils.config import Config


def new_config() -> Config:
    config = Config()
    config["wandb"]["enable"] = True
    config["wandb"]["project"] = "entropic-hp-tuning"
    return config


def main():  # pylint: disable=too-many-locals
    config = None
    model_types = ("entropic_gcn",)
    model_depths = (16, 32, 64, 512, 1024)
    temperatures = (1e-1, 1e0, 1e1)
    weights = (1e-2, 1e-1, 1e0, 1e1)

    combinations = itertools.product(model_types, model_depths, temperatures, weights)

    for model_type, model_depth, temperature, weight in combinations:
        if config:
            del config
        config = new_config()
        config["note"] = f"hp-t_{model_type}-d_{model_depth}-t_{temperature}-w_{weight}"

        # model
        config["model_type"] = model_type
        config["model_parameters"][model_type]["depth"] = model_depth
        config["model_parameters"][model_type]["temperature"] = temperature
        config["model_parameters"][model_type]["weight"] = weight

        agent = BaseTrainer()
        agent.train()
        del agent


if __name__ == "__main__":
    main()
