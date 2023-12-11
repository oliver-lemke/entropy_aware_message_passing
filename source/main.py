from tester.base_tester import BaseTester
from trainer.base_trainer import BaseTrainer
from utils.config import Config

config = Config()


def main():
    if config["run_type"] == "train":
        agent = BaseTrainer()
        agent.train()
    elif config["run_type"] == "test":
        tester = BaseTester()
        tester.test_energy_per_layer()


if __name__ == "__main__":
    main()
