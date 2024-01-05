# pylint: disable-all
from __future__ import annotations

import json
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
api = wandb.Api()
PATH = "gnn_dl/energy_over_layers"
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


def get_artifact_json(run: Run, cache_path: str, step=50) -> pd.DataFrame:
    table_name = f"entropy_over_layersstep{step:04d}_table"
    tag = f"run-{run.name}-{table_name}:v0"
    artifact_path = f"{PATH}/{tag}"
    artifact = api.artifact(artifact_path)
    cache_root = os.path.dirname(cache_path)
    artifact_dir = artifact.download(cache_root + "/artifacts")
    artifact_path_local = os.path.join(
        artifact_dir, "entropy_over_layers", f"step{step:04d}_table.table.json"
    )
    with open(artifact_path_local, "rb") as f:
        artifact_json = json.load(f)
    artifact_df = pd.DataFrame(
        data=artifact_json["data"], columns=artifact_json["columns"]
    )
    return artifact_df


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


def build_pandas_row(
    run: RunsTuple,
) -> pd.DataFrame:
    name, values = run
    config = values["config"]
    energy_df = values["energy_df"]
    model_type = config["model_type"]
    value_dict = {
        "name": name,
        "model_type": strip_string_from_right(model_type, "_gcn"),
        "energy_df": energy_df,
    }
    value_dict = {k: [v] for k, v in value_dict.items()}
    return pd.DataFrame.from_dict(value_dict)


def cache_runs(runs: list[Run], cache_path: str) -> RunsDict:
    runs_dict: RunsDict = {}
    for idx, run in enumerate(runs):
        print(f"{idx=}")
        energy_df = get_artifact_json(run, cache_path)[:-1]
        inner_dict = {
            "state": run.state,
            "config": run.config,
            "history": run.history(),
            "energy_df": energy_df,
        }
        runs_dict[run.id] = inner_dict

    with open(cache_path, "wb") as file:
        pickle.dump(runs_dict, file)
    return runs_dict


def load_runs(name: str, load_if_cached: bool = True) -> RunsDict:
    cache_path = compute_cache_name(name)
    if os.path.exists(cache_path) and load_if_cached:
        with open(cache_path, "rb") as file:
            runs = pickle.load(file)
    else:
        runs = api.runs(name)
        runs = cache_runs(runs, cache_path)
    return runs


def load_df(
    name: str,
    predicate_function: (Callable[[RunsTuple], bool] | None) = None,
) -> pd.DataFrame:
    runs = load_runs(name)
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
    df = load_df(PATH)
    df = unnest_dataframe(df, "energy_df")

    # break into two
    g2_filter = df["model_type"] == "g2"
    df_lower = df[~g2_filter]
    df_higher = df[g2_filter]

    # Setting the aesthetic style of the plots
    sns.set_style("whitegrid")
    # Setting the font scale for larger text
    sns.set_context("notebook", font_scale=1.5)

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    custom_palette = {
        "pairnorm": "blue",
        "g2": "darkorange",
        "entropic": "green",
        "basic": "red",
    }

    # Plot the data on the two subplots
    sns.lineplot(
        data=df_higher,
        x="Layer",
        y="Energy",
        hue="model_type",
        palette=custom_palette,
        linewidth=3.5,
        ax=ax1,
    )
    sns.lineplot(
        data=df_lower,
        x="Layer",
        y="Energy",
        hue="model_type",
        palette=custom_palette,
        linewidth=3.5,
        ax=ax2,
    )

    # This makes the plot look better by removing the top and bottom spines
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    ax1.set_xlim(0, 64)
    ax1.set_ylim(6, 4000)
    ax1.set_ylabel("")
    ax2.set_xlim(0, 64)
    ax2.set_ylim(0, 6)

    ax1.legend()
    ax2.legend()

    # Add diagonal lines to indicate the broken axis
    d = 0.015  # Size of diagonal lines
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
