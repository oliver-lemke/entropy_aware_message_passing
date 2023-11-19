from models.basic_gcn import BasicGCN
from physics.physics import Entropy
from utils.config import Config
from datasets import DatasetFactory

import torch
import torch_geometric as tg

config = Config()


class EntropicGCN(BasicGCN):
    def __init__(self, *args, **kwargs):
        """Entropic Wrapper"""
        super().__init__(*args, **kwargs)

        params = config["model_parameters"]["entropic_gcn"]
        
        A = torch.randint(2, size=(2708, 2708), dtype=torch.float32)
        self.entropy = Entropy(params["temperature"], A=A)


    def forward(self, data):
        """Adjust forward pass to include gradient ascend on the entropy

        Args:
            data (_type_): _description_
        """

        embedding = super().forward(data)

        embedding += self.entropy.gradient_entropy(embedding)

        return embedding
