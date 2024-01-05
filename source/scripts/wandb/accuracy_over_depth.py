from __future__ import annotations

import os
import pickle
from typing import Any, Callable, NewType

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from utils.config import Config
from wandb.apis.public import Run

system_config = Config()

RunsDictEntry = NewType("RunsDictEntry", dict[str, Any])
RunsDict = NewType("RunsDict", dict[str, RunsDictEntry])
RunsTuple = NewType("RunsTuple", tuple[str, RunsDictEntry])


def pre_filters(run: dict[str, Any]) -> bool:
    pred1 = run["state"] == "crashed"
    return any((pred1,))


def base_predicate(run: RunsTuple) -> bool:
    _, values = run
    if pre_filters(values):
        return False
    return True


def strip_string_from_right(original, to_strip):
    """
    Strips the specified string from the right end of the original string, if it exists.

    Parameters:
    original (str): The original string.
    to_strip (str): The string to be stripped from the right end.

    Returns:
    str: The modified string with the specified string stripped from the right, if it was present.
    """
    if original.endswith(to_strip):
        return original[: -len(to_strip)]
    return original


def compute_cache_name(wandb_name: str) -> str:
    name = wandb_name.replace("/", "-")
    name += ".pkl"
    cache_path = os.path.join(system_config.get_subpath("cache"), name)
    return cache_path


def build_pandas_row(
    run: RunsTuple,
) -> pd.DataFrame:
    name, values = run
    config, history = values["config"], values["history"]
    model_type = config["model_type"]
    value_dict = {
        "name": name,
        "model_type": strip_string_from_right(model_type, "_gcn"),
        "depth": config["model_parameters"][model_type]["depth"],
        "best_accuracy": history["val/accuracy"].max(),
    }
    value_dict = {k: [v] for k, v in value_dict.items()}
    return pd.DataFrame.from_dict(value_dict)


def cache_runs(runs: list[Run], cache_path: str) -> RunsDict:
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


def load_runs(name: str, wandb_path: str, load_if_cached: bool = True) -> RunsDict:
    api = wandb.Api()
    cache_path = compute_cache_name(wandb_path)
    if os.path.exists(wandb_path) and load_if_cached:
        with open(cache_path, "rb") as file:
            runs = pickle.load(file)
    else:
        runs = api.runs(name)
        runs = cache_runs(runs, cache_path)
    return runs


def load_df(
    name: str,
    wandb_path: str,
    predicate_function: (Callable[[RunsTuple], bool] | None) = None,
) -> pd.DataFrame:
    runs = load_runs(name, wandb_path)
    if predicate_function is not None:
        runs = [
            run
            for run in runs.items()
            if (base_predicate(run) and predicate_function(run))
        ]
    else:
        runs = [run for run in runs.items() if base_predicate(run)]
    df = pd.concat([build_pandas_row(run) for run in runs], axis=0)
    return df


def main() -> None:
    PATH = "gnn_dl/Cora"
    df = load_df(PATH, PATH)

    # Setting the aesthetic style of the plots
    sns.set_style("whitegrid")
    # Setting the font scale for larger text
    sns.set_context("notebook", font_scale=1.5)
    custom_palette = {
        "pairnorm": "blue",
        "g2": "darkorange",
        "entropic": "green",
        "basic": "red",
    }

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="depth", y="best_accuracy", hue="model_type", linewidth=3.5)
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xlim(0, 64)
    plt.show()


if __name__ == "__main__":
    main()
