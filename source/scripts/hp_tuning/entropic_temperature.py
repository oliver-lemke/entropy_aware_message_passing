import itertools

from trainer.base_trainer import BaseTrainer
from utils.config import Config


def new_config() -> Config:
    config = Config()
    config["model_type"] = "entropic_gcn"
    config["wandb"]["enable"] = True
    config["wandb"]["extended"] = True
    config["wandb"]["project"] = "entropic-temperature-extended"
    return config


def main():
    model_type = "entropic_gcn"
    model_depths = (64,)
    temp_learns = (False,)
    temps = (1e1,)
    weight_learns = (False,)
    weights = (1,)
    normalizes = (True,)
    repetitions = 1

    combinations = itertools.product(
        model_depths,
        temp_learns,
        temps,
        weight_learns,
        weights,
        normalizes,
        list(range(repetitions)),
    )

    for (
        model_depth,
        temp_learn,
        temp,
        weight_learn,
        weight,
        normalize,
        repetition,
    ) in combinations:
        config = new_config()
        config["note"] = f"hp-d_{model_depth}-t_{temp}-w_{weight}_r{repetition}"

        # model
        config["model_type"] = model_type
        config["model_parameters"][model_type]["depth"] = model_depth
        config["model_parameters"][model_type]["temperature"]["learnable"] = temp_learn
        config["model_parameters"][model_type]["temperature"]["value"] = temp
        config["model_parameters"][model_type]["weight"]["learnable"] = weight_learn
        config["model_parameters"][model_type]["weight"]["value"] = weight
        config["model_parameters"][model_type]["normalize_energies"] = normalize

        agent = BaseTrainer()
        agent.train()
        del agent


if __name__ == "__main__":
    main()
