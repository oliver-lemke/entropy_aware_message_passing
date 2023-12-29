from __future__ import annotations

import os
import pickle
from typing import Any, NewType, Callable

import pandas as pd
import wandb
from utils.config import Config
from wandb.apis.public import Run
import matplotlib.pyplot as plt
import seaborn as sns

system_config = Config()
cache_path = os.path.join(system_config.get_subpath("cache"), "wandb_runs.pkl")

RunsDictEntry = NewType("RunsDictEntry", dict[str, Any])
RunsDict = NewType("RunsDict", dict[str, RunsDictEntry])
RunsTuple = NewType("RunsTuple", tuple[str, RunsDictEntry])


def pre_filters(run: dict[str, Any]) -> bool:
    pred1 = run["state"] == "crashed"
    return any((pred1,))


def base_predicate(run: RunsTuple, *args, depth: int = 64, **kwargs) -> bool:
    _, values = run
    config = values["config"]
    if pre_filters(values):
        return False
    pred1 = config["model_type"] == "entropic_gcn"
    pred2 = not config["model_parameters"]["entropic_gcn"]["temperature"]["learnable"]
    return all((pred1, pred2))


def build_pandas_row(run: RunsTuple, *args, **kwargs) -> pd.DataFrame:
    name, values = run
    config, history = values["config"], values["history"]
    value_dict = {"name": name, "depth": config["model_parameters"]["entropic_gcn"]["depth"],
                  "temperature": config["model_parameters"]["entropic_gcn"]["temperature"]["value"],
                  "weight": config["model_parameters"]["entropic_gcn"]["weight"]["value"],
                  "best_metric": history["val/total_loss"].min(), "best_accuracy": history["val/accuracy"].max(), }
    value_dict = {k: [v] for k, v in value_dict.items()}
    return pd.DataFrame.from_dict(value_dict)


def cache_runs(runs: list[Run]) -> RunsDict:
    runs_dict: RunsDict = {}
    for idx, run in enumerate(runs):
        print(f"{idx=}")
        inner_dict = {"state": run.state, "config": run.config, "history": run.history(), }
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
        runs = cache_runs(runs)
    return runs


def load_df(name: str, predicate_function: (Callable[[RunsTuple], bool] | None) = None) -> pd.DataFrame:
    runs = load_runs(name)
    if predicate_function is not None:
        runs = [run for run in runs.items() if predicate_function(run)]
    else:
        runs = runs.items()
    df = pd.concat([build_pandas_row(run) for run in runs], axis=0)
    return df


def heatmap_by_depth(depth: int) -> None:
    if depth is None:
        df = load_df("gnn_dl/entropic-hp-tuning")
    else:
        def predicate_function(run: RunsTuple) -> bool:
            return run[1]["config"]["model_parameters"]["entropic_gcn"]["depth"] == depth

        df = load_df("gnn_dl/entropic-hp-tuning", predicate_function)
    df = df.drop("name", axis=1)
    acc_avg = df.groupby(by=["temperature", "weight"]).mean()["best_accuracy"]
    acc_avg = acc_avg.unstack(level=-1)

    # Converting index and columns to numeric for proper sorting
    acc_avg.index = pd.to_numeric(acc_avg.index)
    acc_avg.columns = pd.to_numeric(acc_avg.columns)
    acc_avg = acc_avg.sort_index().sort_index(axis=1)

    # Creating the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(acc_avg, annot=True, cmap='coolwarm', cbar_kws={'label': 'Best Accuracy'}, annot_kws={"size": 10})
    plt.title(f'Heatmap of Best Accuracy, Depth = {depth}', fontsize=14)
    plt.xlabel('Weight', fontsize=12)
    plt.ylabel('Temperature', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def main():
    DEPTHS = [16, 32, 64, None]
    for depth in DEPTHS:
        heatmap_by_depth(depth)


if __name__ == "__main__":
    main()
