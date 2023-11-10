from models.model_utils import conv_blocks, fusion_blocks, transform_blocks
from utils.config import Config
from utils.logs import Logger

config = Config()
logger = Logger()


class BlockFactory:
    """
    Return the corresponding block class depending on the config
    """

    def __init__(self):
        """
        Each entry should have 1 attribute:
        (1) A function for the model constructor
        """
        self.conv_blocks = conv_blocks.BLOCK_DICT
        self.fusion_blocks = fusion_blocks.BLOCK_DICT
        self.transform_blocks = transform_blocks.BLOCK_DICT

    def get_conv_block(self):
        """
        Returns conv block class
        """
        conv_block_name = config["conv_block_args"]["block_type"]
        return self.conv_blocks[conv_block_name]

    def get_fusion_block(self):
        """
        Returns fusion block class
        """
        fusion_block_name = config["fusion_block_args"]["block_type"]
        return self.fusion_blocks[fusion_block_name]

    def get_transform_block(self):
        """
        Returns transform block class
        """
        transform_block_name = config["transform_block_args"]["block_type"]
        return self.transform_blocks[transform_block_name]
