import os.path

import torch

from models.basic_gcn import BasicGCN
from models.entropic_gcn import EntropicGCN
from models.hrnet_gcn import HRNetGCN
from models.g2 import G2_GNN
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class ModelFactory:
    def __init__(self):
        """
        Each entry should have 1 attribute:
        (1) A function for the model constructor
        """
        self.models = {
            "basic_gcn": BasicGCN,
            "entropic_gcn": EntropicGCN,
            "hrnet_gcn": HRNetGCN,
            "g2": G2_GNN,
        }

    def get_model(self, *args, **kwargs):
        """
        Instantiates model from available pool
        :return: return the specified model (torch.nn.Module), as well as transforms
        """
        model_type = config["model_type"]
        Model = self.models[model_type]
        logger.info(f"Loading Model {model_type}: {Model}")

        model = Model(*args, **kwargs)

        # load pretrained weights if applicable
        model_parameters = config["model_parameters"][model_type]
        weights_path_extension = model_parameters["pretrained_weights"]
        if weights_path_extension is not None:
            logger.info(f"Loading pretrained weights {weights_path_extension}!")
            weights_path = os.path.join(
                config.get_subpath("pretrained_weights"),
                weights_path_extension,
                config["save_names"]["weights"],
            )
            pretrained_dict = torch.load(weights_path)
            model.load_state_dict(pretrained_dict)
        else:
            logger.info("Training model from scratch!")

        # compute a config dict which specifies some transformations for the inputs
        # (might be different for different models)
        transform_config = None
        logger.debug(transform_config)
        return model, transform_config
