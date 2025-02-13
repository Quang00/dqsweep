import itertools
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================
LOG_SCALE_PARAMS = {
    "length",
    "T1",
    "T2",
    "single_qubit_gate_time",
    "two_qubit_gate_time",
}


# =============================================================================
# Helper Functions
# =============================================================================
def truncate_param(name: str, n: int = 3) -> str:
    """Truncates a parameter name to improve readability in plots.

    Args:
        name (str): Parameter name to truncate.
        n (int, optional): Number of tokens to keep. Defaults to 3.

    Returns:
        str: Truncated parameter name.
    """
    return " ".join(name.split("_")[:n])


def create_subdir(
    directory: str, experiment: str, sweep_params: Union[list, str]
) -> str:
    """Creates a structured subdirectory for storing experiment results.

    Args:
        directory (str): The main results directory (e.g., "results").
        experiment (str): The experiment name (e.g., "pingpong").
        sweep_params (list | str): List of swept parameters (or a string).

    Returns:
        str: Path to the created experiment-specific directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    if isinstance(sweep_params, str):
        sweep_params = sweep_params.split(",")

    sweep_params = [param.strip() for param in sweep_params]
    param_names = (
        "_".join(truncate_param(param, n=1) for param in sweep_params)
        if sweep_params
        else "default"
    )
    experiment_dir = os.path.join(directory, f"{experiment}_{param_names}")
    cpt = 1
    new_subdir = experiment_dir

    while os.path.exists(new_subdir):
        new_subdir = f"{experiment_dir}_{cpt}"
        cpt += 1

    os.makedirs(new_subdir)

    return new_subdir


def parse_range(range_str: str, param_name: str) -> np.ndarray:
    """Parses a range string and returns a numpy array of values.

    Uses logarithmic scaling if the parameter is in LOG_SCALE_PARAMS.

    Args:
        range_str (str): Range in format "start,end,points".
        param_name (str): The name of the parameter being parsed.

    Returns:
        np.ndarray: Array of evenly spaced values.
    """
    try:
        start, end, points = map(float, range_str.split(","))
        if param_name in LOG_SCALE_PARAMS:
            return np.logspace(np.log10(start), np.log10(end), int(points))
        return np.linspace(start, end, int(points))
    except ValueError:
        raise ValueError("Invalid range format. Use 'start,end,points'.") from None


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_parameter_metric_correlation(
    df: pd.DataFrame,
    sweep_params: list,
    metric_cols: list,
    output_dir: str,
    experiment: str,
):
    """Computes and plots a bar chart for parameter-performance correlation.

    Args:
        df (pd.DataFrame): Dataframe containing parameter values and metrics.
        sweep_params (list): List of parameters being swept.
        metric_cols (list): List of performance metrics.
        output_dir (str): Directory to save the plot.
        experiment (str): Experiment name.
    """
    corr_data = df[sweep_params + metric_cols].corr()
    corr_subset = corr_data.loc[sweep_params, metric_cols]

    x = np.arange(len(sweep_params))
    width = 0.35

    _, ax = plt.subplots(figsize=(12, len(sweep_params) * 0.8 + 4))
    for i, metric in enumerate(metric_cols):
        ax.barh(x + (i * width), corr_subset[metric], width, label=metric)

    ax.set_yticks(x + width / 2)
    ax.set_yticklabels(sweep_params, fontsize=14)
    ax.set_xlabel("Correlation Coefficient", fontsize=14)
    ax.set_title(
        f"{experiment.capitalize()} Parameter-Performance Correlation",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(title="Performance Metrics", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{experiment}_param_metric_correlation.png")
    plt.savefig(filename, dpi=1000)
    plt.close()
    print(f"Saved correlation bar chart to {filename}")


def plot_combined_3d_surfaces(
    df: pd.DataFrame,
    sweep_params: list,
    param_range_dict: dict,
    output_dir: str,
    experiment: str,
    epr_rounds: int,
):
    """Generates a single figure containing all 3D surface plots.

    Args:
        df (pd.DataFrame): Dataframe containing experiment results.
        sweep_params (list): List of swept parameters.
        param_range_dict (dict): Dictionary mapping parameters to their ranges.
        output_dir (str): Directory to save the generated figure.
        experiment (str): Name of the experiment.
        epr_rounds (int): Number of epr rounds.
    """
    pairs = list(itertools.combinations(sweep_params, 2))
    fig, axes = plt.subplots(
        2, len(pairs), figsize=(12 * len(pairs), 16), subplot_kw={"projection": "3d"}
    )
    fig.subplots_adjust(wspace=1, hspace=1)

    if len(pairs) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for row, metric_name, cmap_style in zip(
        range(2),
        ["Average Fidelity (%)", "Average Simulation Time (ms)"],
        ["magma", "viridis"],
    ):
        for col, (p, q) in enumerate(pairs):
            ax = axes[row, col]

            # Generate pivot table for metric
            pivot = df.pivot_table(
                index=p, columns=q, values=metric_name, aggfunc="mean"
            )
            metric_matrix = pivot.values

            # Ensure correct meshgrid
            x = param_range_dict[p]
            y = param_range_dict[q]
            x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")

            # Transpose metric_matrix to align with meshgrid
            surf = ax.plot_surface(
                x_mesh, y_mesh, metric_matrix.T, cmap=cmap_style, edgecolor="none"
            )
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
            ax.set_xlabel(truncate_param(q), fontsize=14)
            ax.set_ylabel(truncate_param(p), fontsize=14)
            if experiment == "pingpong":
                ax.set_title(
                    f"{truncate_param(p)} vs {truncate_param(q)} with {epr_rounds}",
                    fontsize=16,
                    fontweight="bold",
                )
            else:
                ax.set_title(
                    f"{truncate_param(p)} vs {truncate_param(q)}",
                    fontsize=16,
                    fontweight="bold",
                )

    plt.tight_layout()
    filename = os.path.join(output_dir, f"{experiment}_3d_surfaces.png")
    plt.savefig(filename, dpi=1000)
    plt.close()
    print(f"Saved all 3D surface plots to {filename}")


def plot_combined_heatmaps(
    df: pd.DataFrame,
    sweep_params: list,
    param_range_dict: dict,
    output_dir: str,
    experiment: str,
    epr_rounds: int,
):
    """Generates a single figure containing all 2D heatmaps.

    Args:
        df (pd.DataFrame): Dataframe containing experiment results.
        sweep_params (list): List of swept parameters.
        param_range_dict (dict): Dictionary mapping parameters to their ranges.
        output_dir (str): Directory to save the generated figure.
        experiment (str): Name of the experiment.
        epr_rounds (int): Number of epr rounds.
    """
    pairs = list(itertools.combinations(sweep_params, 2))
    fig, axes = plt.subplots(2, len(pairs), figsize=(12 * len(pairs), 16))
    fig.subplots_adjust(wspace=1, hspace=1)

    if len(pairs) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for row, metric_name, cmap_style in zip(
        range(2),
        ["Average Fidelity (%)", "Average Simulation Time (ms)"],
        ["magma", "viridis"],
    ):
        for col, (p, q) in enumerate(pairs):
            ax = axes[row, col]
            # Generate pivot table for metric
            pivot = df.pivot_table(
                index=p, columns=q, values=metric_name, aggfunc="mean"
            )
            metric_matrix = pivot.values

            im = ax.imshow(
                metric_matrix,
                extent=[
                    param_range_dict[q][0],
                    param_range_dict[q][-1],  # X-axis
                    param_range_dict[p][0],  # Y-axis
                    param_range_dict[p][-1],
                ],
                origin="lower",
                aspect="auto",
                cmap=cmap_style,
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(metric_name, fontsize=14)
            ax.set_xlabel(truncate_param(q), fontsize=14)
            ax.set_ylabel(truncate_param(p), fontsize=14)
            ax.set_title(
                (
                    f"{truncate_param(p)} vs {truncate_param(q)} with {epr_rounds} hops"
                    if experiment == "pingpong"
                    else f"{truncate_param(p)} vs {truncate_param(q)}"
                ),
                fontsize=16,
                fontweight="bold",
            )

    plt.tight_layout()
    filename = os.path.join(output_dir, f"{experiment}_heatmaps.png")
    plt.savefig(filename, dpi=1000)
    plt.close()
    print(f"Saved all heatmaps to {filename}")
