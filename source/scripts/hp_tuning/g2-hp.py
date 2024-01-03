import itertools

from trainer.base_trainer import BaseTrainer
from utils.config import Config


def new_config() -> Config:
    config = Config()
    config["model_type"] = "g2"
    config["wandb"]["enable"] = True
    config["wandb"]["extended"] = False
    config["wandb"]["project"] = "g2-hp"
    return config


def main():
    model_type = "g2"
    repetitions = 1
    model_depths = (16, 32, 64)
    use_gg_convs = (True, False)
    conv_types = ("GCN", "GAT")
    ps = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

    combinations = itertools.product(
        list(range(repetitions)), model_depths, use_gg_convs, conv_types, ps
    )

    for (
        repetition,
        model_depth,
        use_gg_conv,
        conv_type,
        p,
    ) in combinations:
        config = new_config()
        config["note"] = (
            f"hp-d_{model_depth}-gg_{use_gg_conv}-c_{conv_type}-"
            f"p_{p}-r_{repetition}"
        )

        # model
        config["model_type"] = model_type
        config["model_parameters"][model_type]["depth"] = model_depth
        config["model_parameters"][model_type]["use_gg_conv"] = use_gg_conv
        config["model_parameters"][model_type]["conv_type"] = conv_type
        config["model_parameters"][model_type]["p"] = p

        agent = BaseTrainer()
        agent.train()
        del agent


if __name__ == "__main__":
    main()
