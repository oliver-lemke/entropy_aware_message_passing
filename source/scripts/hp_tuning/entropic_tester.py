import itertools
import time

from tester.base_tester import BaseTester
from utils.config import Config


def new_config() -> Config:
    config = Config()
    config["model_type"] = "entropic_gcn"
    config["run_type"] = "test"
    config["wandb"]["enable"] = True
    config["wandb"]["extended"] = False
    config["wandb"]["project"] = "entropic-hp-tester"
    return config


def main():
    model_type = "entropic_gcn"
    temps = (1e-2, 1e-1, 1e0, 1e1, 1e2)
    weights = (1e-2, 1e-1, 5e-1, 1e0, 1e1, 1e2)
    normalize_energies = (True, False)

    combinations = itertools.product(temps, weights, normalize_energies)

    for temp, weight, normalize in combinations:
        config = new_config()
        config["note"] = f"tester-hp-t_{temp}-w_{weight}-n_{normalize}"

        # model
        config["model_parameters"][model_type]["temperature"]["value"] = temp
        config["model_parameters"][model_type]["weight"]["value"] = weight
        config["model_parameters"][model_type]["normalize_energies"] = normalize

        tester = BaseTester()
        tester.test_energy_per_layer()
        del tester
        time.sleep(5)


if __name__ == "__main__":
    main()
