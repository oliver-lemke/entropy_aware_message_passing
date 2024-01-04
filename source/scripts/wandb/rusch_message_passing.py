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
cache_path = os.path.join(system_config.get_subpath("cache"), "wandb_runs.pkl")

RunsDictEntry = NewType("RunsDictEntry", dict[str, Any])
RunsDict = NewType("RunsDict", dict[str, RunsDictEntry])
RunsTuple = NewType("RunsTuple", tuple[str, RunsDictEntry])


def pre_filters(run: dict[str, Any]) -> bool:
    pred1 = run["state"] == "crashed"
    return any((pred1,))


def base_predicate(run: RunsTuple) -> bool:
    _, values = run
    config = values["config"]
    if pre_filters(values):
        return False
    pred1 = config["model_type"] == "entropic_gcn"
    pred2 = not config["model_parameters"]["entropic_gcn"]["temperature"]["learnable"]
    return all((pred1, pred2))


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


def build_pandas_row(
    run: RunsTuple,
) -> pd.DataFrame:
    name, values = run
    config, history = values["config"], values["history"]
    model_type = config["tester"]["model_type"]
    df = history.T.loc[["depth", f"energy/{model_type}"]].T
    depth_energy_df = df[~df.isna().any(axis=1)]
    depth_energy_df.columns = ["depth", "energy"]
    value_dict = {
        "model_type": strip_string_from_right(model_type, "_gcn"),
        "df": depth_energy_df.iloc[1:],
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
        runs = cache_runs(runs)
    return runs


def load_df(
    name: str, predicate_function: (Callable[[RunsTuple], bool] | None) = None
) -> pd.DataFrame:
    runs = load_runs(name)
    if predicate_function is not None:
        runs = [run for run in runs.items() if predicate_function(run)]
    else:
        runs = runs.items()
    df = pd.concat([build_pandas_row(run) for run in runs], axis=0)
    return df


def unnest_dataframe(df, column_name="df"):
    unnested_dfs = []
    for index, row in df.iterrows():
        inner_df = row[column_name]
        series = row.drop(column_name, axis=0)
        for idx, (k, v) in enumerate(series.items()):
            inner_df.insert(idx, k, v)
        unnested_dfs.append(inner_df)

    # Combine all unnested DataFrames
    unnested_df = pd.concat(unnested_dfs).reset_index(drop=True)
    return unnested_df


def main() -> None:
    PATH = "gnn_dl/testing"
    df = load_df(PATH)
    df = unnest_dataframe(df, column_name="df")

    # Setting the aesthetic style of the plots
    sns.set_style("whitegrid")
    # Setting the font scale for larger text
    sns.set_context("notebook", font_scale=1.5)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="depth", y="energy", hue="model_type", linewidth=3.5)
    plt.xlabel("Depth")
    plt.ylabel("Energy")
    plt.legend()
    plt.xlim(0, 1000)
    plt.show()


if __name__ == "__main__":
    main()
