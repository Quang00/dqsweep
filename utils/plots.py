import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.helper import truncate_param


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_heatmap(
    ax,
    df: pd.DataFrame,
    p: str,
    q: str,
    metric: dict,
    params: dict,
    exp: str,
    rounds: int,
):
    """Plot an individual heatmap on a given axes.

    Args:
        ax: Matplotlib Axes object.
        df (pd.DataFrame): Dataframe containing experiment results.
        p (str): Name of the parameter for the y-axis.
        q (str): Name of the parameter for the x-axis.
        metric (dict): Dictionary with keys 'name', 'cmap', and 'file_label'.
        params (dict): Dictionary mapping parameters to their ranges.
        exp (str): Name of the experiment.
        rounds (int): Number of epr rounds.
    """
    pivot_mean = df.pivot_table(index=p, columns=q, values=metric["name"])
    pivot_std = df.pivot_table(index=p, columns=q, values=metric["std"])

    annot = (
        pivot_mean.round(3).astype(str)
        + "\n ±"
        + pivot_std.round(3).astype(str)
    )

    sns.heatmap(
        pivot_mean,
        ax=ax,
        cmap=metric["cmap"],
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": metric["name"]},
        xticklabels=False,
        yticklabels=False,
    )

    x_vals = pivot_mean.columns.values
    y_vals = pivot_mean.index.values

    ax.set_xticks(np.arange(len(x_vals)) + 0.5)
    ax.set_yticks(np.arange(len(y_vals)) + 0.5)

    ax.set_xticklabels([f"{v:.3f}" for v in x_vals], rotation=0, fontsize=12)
    ax.set_yticklabels([f"{v:.3f}" for v in y_vals], rotation=0, fontsize=12)

    ax.invert_yaxis()

    var1 = truncate_param(q)
    var2 = truncate_param(p)

    ax.set_xlabel(var1, fontsize=14)
    ax.set_ylabel(var2, fontsize=14)

    title_suffix = f" with {rounds} hops" if exp == "pingpong" else ""
    ax.set_title(
        f"{exp.capitalize()}: {var2} vs {var1}{title_suffix}",
        fontsize=16,
        fontweight="bold",
    )


def save_single_heatmap(
    metric: dict,
    df: pd.DataFrame,
    pairs: list,
    params: dict,
    exp: str,
    rounds: int,
    dir: str,
) -> None:
    """Generate and save a heatmap for a single metric.

    Args:
        metric (dict): Metric details (keys: 'name', 'cmap', 'file_label').
        df (pd.DataFrame): DataFrame containing experiment results.
        pairs (list): List of parameter pairs generated from sweep_params.
        params (dict): Dictionary mapping parameters to their ranges .
        exp (str): Name of the experiment.
        rounds (int): Number of EPR rounds.
        dir (str): Directory to save the generated figures.
    """
    figsize = (12 * len(pairs), 8)
    fig, axes = plt.subplots(1, len(pairs), figsize=figsize)

    # Ensure axes is iterable.
    if len(pairs) == 1:
        axes = [axes]

    for ax, (p, q) in zip(axes, pairs, strict=False):
        plot_heatmap(ax, df, p, q, metric, params, exp, rounds)

    plt.tight_layout()
    prefix = f"{exp}_heat_{metric['file_label']}"
    suffix = f"{(rounds + 1) // 2}" if exp == "pingpong" else ""
    filename = os.path.join(dir, f"{prefix}_{suffix}.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved {metric['name']} heatmap to {filename}")


def save_combined_heatmaps(
    metrics: list,
    df: pd.DataFrame,
    pairs: list,
    params: dict,
    exp: str,
    rounds: int,
    dir: str,
) -> None:
    """Generate and save a combined figure with heatmaps for all metrics.

    Args:
           metrics (list): Metric details (keys: 'name', 'cmap', 'file_label').
           df (pd.DataFrame): DataFrame containing experiment results.
           pairs (list): List of parameter pairs generated from sweep_params.
           params (dict): Dictionary mapping parameters to their ranges.
           exp (str): Name of the experiment.
           rounds (int): Number of EPR rounds.
           dir (str): Directory to save the generated figures.
    """
    figsize = (12 * len(pairs), 8 * len(metrics))
    fig, axes = plt.subplots(len(metrics), len(pairs), figsize=figsize)

    # Ensure axes is 2D even when there is only one pair.
    if len(pairs) == 1:
        axes = np.array([axes]).reshape(len(metrics), 1)

    for i, metric in enumerate(metrics):
        for j, (p, q) in enumerate(pairs):
            plot_heatmap(axes[i, j], df, p, q, metric, params, exp, rounds)

    plt.tight_layout()
    filename = os.path.join(dir, f"{exp}_heatmaps.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved combined heatmaps to {filename}")


def plot_combined_heatmaps(
    df: pd.DataFrame,
    sweep_params: list,
    params: dict,
    dir: str,
    exp: str,
    rounds: int,
    separate_files: bool = False,
):
    """Generates heatmaps from experiment results.

    If `separate_files` is True, separate figures will be created for each
    metric. Otherwise, a single combined figure is generated.

    Args:
        df (pd.DataFrame): DataFrame containing experiment results.
        sweep_params (list): List of swept parameters.
        params (dict): Dictionary mapping parameters to their ranges.
        dir (str): Directory to save the generated figures.
        exp (str): Name of the experiment.
        rounds (int): Number of EPR rounds.
        separate_files (bool, optional): Whether to generate separate files.
    """
    pairs = list(itertools.combinations(sweep_params, 2))
    metrics = [
        {
            "name": "Average Fidelity",
            "std": "Standard Deviation Fidelity",
            "cmap": "magma",
            "file_label": "fidelity",
        },
        {
            "name": "Average Simulation Time (ms)",
            "std": "Standard Deviation Simulation Time",
            "cmap": "viridis",
            "file_label": "sim_times",
        },
    ]

    if separate_files:
        for metric in metrics:
            save_single_heatmap(metric, df, pairs, params, exp, rounds, dir)
    else:
        save_combined_heatmaps(metrics, df, pairs, params, exp, rounds, dir)
