from trainer.base_trainer import BaseTrainer
from tester.base_tester import BaseTester


def main():
    # agent = BaseTrainer()
    # agent.train()
    tester = BaseTester()
    print(tester.test())


if __name__ == "__main__":
    main()
