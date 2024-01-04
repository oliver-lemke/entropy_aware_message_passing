import itertools
import time

from tester.base_tester import BaseTester
from utils.config import Config


def new_config() -> Config:
    config = Config()
    config["wandb"]["enable"] = True
    config["wandb"]["extended"] = True
    config["wandb"]["project"] = "testing"
    return config


def main():
    run_type = "test"
    model_types = ("pairnorm_gcn",)

    combinations = itertools.product(model_types)

    for (model_type,) in combinations:
        config = new_config()
        config["note"] = f"test-{model_type}"

        # model
        config["run_type"] = run_type
        config["model_type"] = model_type
        config["tester"]["model_type"] = model_type

        tester = BaseTester()
        tester.test_energy_per_layer()
        del tester
        time.sleep(5)


if __name__ == "__main__":
    main()
