import itertools

from trainer.base_trainer import BaseTrainer
from utils.config import Config


def new_config() -> Config:
    config = Config()
    config["wandb"]["enable"] = True
    config["wandb"]["project"] = "hp-tuning"
    return config


def main():  # pylint: disable=too-many-locals
    config = None
    model_types = ("hrnet_gcn",)
    model_depths = (24,)
    branch_everys = (2,)
    block_depths = (2,)
    hidden_dims = (128, 256)
    residuals = (True,)
    conv_block_types = ("basic_gcn",)
    fusion_block_types = ("mean", "max", "simple_conv", "simple_attention")
    transform_block_types = ("id", "full")
    output_block_types = (
        "mean",
        "max",
        "simple_conv",
        "simple_attention",
    )
    attention_types = ("per_node", "per_element")

    combinations = itertools.product(
        model_types,
        model_depths,
        branch_everys,
        block_depths,
        hidden_dims,
        residuals,
        conv_block_types,
        fusion_block_types,
        transform_block_types,
        output_block_types,
        attention_types,
    )

    for (
        model_type,
        model_depth,
        branch_every,
        block_depth,
        hidden_dim,
        residual,
        conv_block_type,
        fusion_block_type,
        transform_block_type,
        output_block_type,
        attention_type,
    ) in combinations:
        if config:
            del config
        config = new_config()
        config["note"] = (
            f"hp-"
            f"t_{model_type}-d_{model_depth}-f_{fusion_block_type}-"
            f"t_{transform_block_type}-o_{output_block_type}"
        )

        # model
        config["model_type"] = model_type
        config["model_parameters"][model_type]["depth"] = model_depth
        if model_type == "hrnet_gcn":
            branches = (i for i in range(0, model_depth, branch_every))
            config["model_parameters"][model_type]["branches"] = list(branches)

        # conv block
        config["conv_block_args"]["block_type"] = conv_block_type
        config["conv_block_args"][conv_block_type] = {}
        config["conv_block_args"][conv_block_type]["depth"] = block_depth
        config["conv_block_args"][conv_block_type]["hidden_dim"] = hidden_dim

        # fusion block
        config["fusion_block_args"]["block_type"] = fusion_block_type
        config["fusion_block_args"][fusion_block_type] = {}
        config["fusion_block_args"][fusion_block_type]["depth"] = block_depth
        config["fusion_block_args"][fusion_block_type]["hidden_dim"] = hidden_dim
        config["fusion_block_args"][fusion_block_type]["residual"] = residual
        config["fusion_block_args"][fusion_block_type][
            "attention_type"
        ] = attention_type

        # transform block
        config["transform_block_args"]["block_type"] = transform_block_type
        config["transform_block_args"][transform_block_type] = {}
        config["transform_block_args"][transform_block_type]["depth"] = block_depth
        config["transform_block_args"][transform_block_type]["hidden_dim"] = hidden_dim
        config["transform_block_args"][transform_block_type]["residual"] = residual

        # output block
        config["output_block_args"]["block_type"] = output_block_type
        config["output_block_args"][output_block_type] = {}
        config["output_block_args"][output_block_type]["depth"] = block_depth
        config["output_block_args"][output_block_type]["hidden_dim"] = hidden_dim
        config["output_block_args"][output_block_type]["residual"] = residual
        config["output_block_args"][output_block_type][
            "attention_type"
        ] = attention_type

        agent = BaseTrainer()
        agent.train()
        del agent


if __name__ == "__main__":
    main()
