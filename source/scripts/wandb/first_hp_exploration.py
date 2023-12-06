from __future__ import annotations

import os
import pickle
from typing import Any, NewType

import pandas as pd
import wandb
from utils.config import Config
from wandb.apis.public import Run

system_config = Config()
cache_path = os.path.join(system_config.get_subpath("cache"), "wandb_runs.pkl")

RunsDict = NewType("RunsDict", dict[Any, dict[str, Any]])


def pre_filters(run) -> bool:
    pred1 = run.state == "crashed"
    return any((pred1,))


def predicate(run: Run) -> bool:
    if pre_filters(run):
        return False
    config = run.config
    pred1 = config["model_type"] == "hrnet_gcn"
    pred2 = config["model_parameters"]["hrnet_gcn"]["depth"] == 16
    return all((pred1, pred2))


def build_pandas_row(name: str, run: RunsDict) -> pd.DataFrame:
    config, history = run["config"], run["history"]
    value_dict = {
        "name": name,
        "depth": config["model_parameters"]["hrnet_gcn"]["depth"],
        "hidden_dim": config["conv_block_args"]["basic_gcn"]["hidden_dim"],
        "best_metric": history["val/total_loss"].min(),
    }
    value_dict = {k: [v] for k, v in value_dict.items()}
    return pd.DataFrame.from_dict(value_dict)


def cache_runs(runs: list[Run]) -> RunsDict:
    runs_dict: RunsDict = {}
    for idx, run in enumerate(runs):
        print(f"{idx=}")
        inner_dict = {
            "state": run.state,
            "config": run.config,
            "history": run.history(),
        }
        runs_dict[run.id] = inner_dict

    with open(cache_path, "wb") as file:
        pickle.dump(runs_dict, file)
    return runs_dict


def load_runs(name: str, load_if_cached: bool = True) -> RunsDict:
    api = wandb.Api()
    if os.path.exists(cache_path) and load_if_cached:
        with open(cache_path, "rb") as file:
            runs = pickle.load(file)
    else:
        runs = api.runs(name)
        runs = [run for run in runs if predicate(run)]
        runs = cache_runs(runs)
    return runs


def main():
    runs = load_runs("gnn_dl/hp-tuning")
    df = pd.concat([build_pandas_row(*run) for run in runs.items()], axis=0)
    pass


if __name__ == "__main__":
    main()
