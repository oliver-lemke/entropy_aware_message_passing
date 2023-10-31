import os.path

import torch

from models.model_template import ModelTemplate
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class ModelFactory:
    def __init__(self):
        """
        Each entry should have 4 attributes:
        (1) A function for the model constructor
        (2) kwargs for function (1)
        (3) A name under which to find pretrained weights
        (4) A model type description which specifies which the type of model. Used when extending the #channels
        """
        self.basic_models = {
            "model_template": (
                ModelTemplate,
                {},
                "default_weights",
                "normal_cnn",
            )
        }

    def get_model(self, model_type: str, *args, **kwargs):
        """
        Instantiates model from available pool
        :param model_type: specifies the type of model
        :param args: to cath additional args
        :param kwargs: to cath additional args
        :return: return the specified model (torch.nn.Module), as well as transforms
        """

        (
            model_constructor,
            kwargs,
            weights_path_extension,
            model_class,  # pylint: disable=unused-variable
        ) = self.basic_models[model_type]
        model = model_constructor(*args, **kwargs)

        pretrained_weights_path = config["pretrained_weights"]
        if not (
            pretrained_weights_path is None or pretrained_weights_path.strip() == ""
        ):
            weights_path = os.path.join(
                pretrained_weights_path, weights_path_extension, "weights.pth"
            )
            pretrained_dict = torch.load(weights_path)
            model.load_state_dict(pretrained_dict)

        model_flags = config["model_flags"]  # pylint: disable=unused-variable
        # Change model architecture according to model_flags
        id(model)

        # compute a config dict which specifies some transformations for the inputs
        # (might be different for different models)
        transform_config = None
        return model, transform_config
