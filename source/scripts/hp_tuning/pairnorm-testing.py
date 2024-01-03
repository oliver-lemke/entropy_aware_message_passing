import itertools
import time

from tester.base_tester import BaseTester
from utils.config import Config


def new_config() -> Config:
    config = Config()
    config["wandb"]["enable"] = True
    config["wandb"]["extended"] = True
    config["wandb"]["project"] = "pairnorm-test"
    return config


def main():
    run_type = "test"
    model_type = "pairnorm_gcn"
    modes = ("PN", "PN-SI", "PN-SCS", None)

    combinations = itertools.product(modes)

    for (mode,) in combinations:
        config = new_config()
        config["note"] = f"pairnorm-test-m_{mode}"

        # model
        config["run_type"] = run_type
        config["model_type"] = model_type
        config["tester"]["model_type"] = model_type
        config["model_parameters"][model_type]["norm_mode"] = mode

        tester = BaseTester()
        tester.test_energy_per_layer()
        del tester
        time.sleep(10)


if __name__ == "__main__":
    main()
