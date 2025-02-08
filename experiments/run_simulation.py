#!/usr/bin/env python3
"""
Run Simulation, Parameter Sweep, and Plot Results
----------------------------------------------------

This script runs a specified quantum network experiment (e.g., cnot, teleportation, pingpong)
while sweeping one or more configuration parameters. For each parameter combination the simulation
is executed (averaging over a number of experiments) and the average fidelity and simulation time
are computed. The results are collected into a pandas DataFrame, saved as a CSV file, and then
the following plots are generated:
  - 2D heatmaps for each performance metric for every unique pair of swept parameters.
    The colormap is chosen based on the metric:
      • If the metric contains "Fidelity", 'magma' is used.
      • If the metric contains "Simulation", 'cividis' is used.
  - 3D surface plots for each unique pair of swept parameters, using the same aggregated data.
    The colormap is similarly chosen based on the metric.
  - A parameter–performance correlation heatmap (showing the correlation between each input parameter
    and each performance metric).

Parameters:
  --config         Path to the configuration YAML file.
  --experiment     Experiment to simulate (e.g., cnot, teleportation, pingpong).
  --epr_rounds     Number of EPR rounds (default 10).
  --num_experiments Number of experiments per parameter combination (default 10).
  --sweep_params   Comma-separated list of parameter names to sweep.
  --ranges         One range string per parameter (format: "start,end,points").
"""

import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run

# Import experiment programs.
from nonlocal_cnot import AliceProgram, BobProgram
from pingpong import AlicePingpongTeleportation, BobPingpongTeleportation
from teleportation import AliceTeleportation, BobTeleportation


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


def create_unique_dir(directory: str) -> str:
    """Creates a unique directory if one with the same name exists.

    Args:
        directory (str): Target directory path.

    Returns:
        str: Unique directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        return directory
    counter = 1
    new_dir = f"{directory}_{counter}"
    while os.path.exists(new_dir):
        counter += 1
        new_dir = f"{directory}_{counter}"
    os.makedirs(new_dir)
    return new_dir


def parse_range(range_str: str) -> np.ndarray:
    """Parses a range string and returns a numpy array of values.

    Args:
        range_str (str): Range in format "start,end,points".

    Returns:
        np.ndarray: Array of evenly spaced values.
    """
    try:
        start, end, points = map(float, range_str.split(","))
        return np.linspace(start, end, int(points))
    except ValueError:
        raise ValueError("Invalid range format. Use 'start,end,points'.") from None


# =============================================================================
# CONSTANTS
# =============================================================================
FIDELITY = "Average Fidelity (%)"
TIME = "Average Simulation Time (ms)"


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

    fig, ax = plt.subplots(figsize=(12, len(sweep_params) * 0.8 + 4))
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
):
    """Generates a single figure containing all 3D surface plots.

    This function creates a 3D surface plot for each unique pair of swept
    parameters, displaying the results for both Average Fidelity and
    Simulation Time in a single figure.

    Args:
        df (pd.DataFrame): Dataframe containing experiment results.
        sweep_params (list): List of swept parameters.
        param_range_dict (dict): Dictionary mapping parameters to their ranges.
        output_dir (str): Directory to save the generated figure.
        experiment (str): Name of the experiment.
    """
    pairs = list(itertools.combinations(sweep_params, 2))
    fig, axes = plt.subplots(
        2, len(pairs), figsize=(24, 16), subplot_kw={"projection": "3d"}
    )
    fig.subplots_adjust(wspace=1, hspace=1)

    # Ensure axes are correctly formatted when only one pair exists
    if len(pairs) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for row, metric_name, cmap_style in zip(
        range(2), [FIDELITY, TIME], ["magma", "viridis"]
    ):
        for col, (p, q) in enumerate(pairs):
            ax = axes[row, col]
            pivot = df.pivot_table(
                index=p, columns=q, values=metric_name, aggfunc=np.mean
            )
            metric_matrix = pivot.values
            x = param_range_dict[p]
            y = param_range_dict[q]
            x_mesh, y_mesh = np.meshgrid(x, y)
            surf = ax.plot_surface(
                x_mesh, y_mesh, metric_matrix, cmap=cmap_style, edgecolor="none"
            )
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
            ax.set_xlabel(truncate_param(q), fontsize=14)
            ax.set_ylabel(truncate_param(p), fontsize=14)
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
):
    """Generates a single figure containing all 2D heatmaps.

    This function creates heatmaps for each unique pair of swept parameters,
    displaying the results for both Average Fidelity and Simulation Time in
    a single figure.

    Args:
        df (pd.DataFrame): Dataframe containing experiment results.
        sweep_params (list): List of swept parameters.
        param_range_dict (dict): Dictionary mapping parameters to their ranges.
        output_dir (str): Directory to save the generated figure.
        experiment (str): Name of the experiment.
    """
    pairs = list(itertools.combinations(sweep_params, 2))
    fig, axes = plt.subplots(2, len(pairs), figsize=(24, 16))
    fig.subplots_adjust(wspace=1, hspace=1)

    # Ensure axes are correctly formatted when only one pair exists
    if len(pairs) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for row, metric_name, cmap_style in zip(
        range(2), [FIDELITY, TIME], ["magma", "viridis"]
    ):
        for col, (p, q) in enumerate(pairs):
            ax = axes[row, col]
            pivot = df.pivot_table(
                index=p, columns=q, values=metric_name, aggfunc=np.mean
            )
            metric_matrix = pivot.values
            im = ax.imshow(
                metric_matrix,
                extent=[
                    param_range_dict[q][0],
                    param_range_dict[q][-1],
                    param_range_dict[p][0],
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
                f"{truncate_param(p)} vs {truncate_param(q)}",
                fontsize=16,
                fontweight="bold",
            )

    plt.tight_layout()
    filename = os.path.join(output_dir, f"{experiment}_heatmaps.png")
    plt.savefig(filename, dpi=1000)
    plt.close()
    print(f"Saved all heatmaps to {filename}")


# =============================================================================
# Parameter Sweep Function
# =============================================================================
def sweep_parameters(
    cfg: StackNetworkConfig,
    epr_rounds: int,
    num_experiments: int,
    sweep_params: list,
    ranges: list,
    experiment: str,
    output_dir: str,
) -> pd.DataFrame:
    """Performs a parameter sweep, runs simulations, and stores results.

    Args:
        cfg (StackNetworkConfig): Network configuration.
        epr_rounds (int): Number of EPR rounds.
        num_experiments (int): Number of experiments per configuration.
        sweep_params (list): Parameters to sweep.
        ranges (list): Ranges for each parameter.
        experiment (str): Experiment name.
        output_dir (str): Directory to save results.

    Returns:
        pd.DataFrame: Dataframe of results.
    """
    os.makedirs(output_dir, exist_ok=True)
    param_ranges = {
        param: parse_range(rng_str) for param, rng_str in zip(sweep_params, ranges)
    }
    combinations = list(
        itertools.product(*(param_ranges[name] for name in sweep_params))
    )
    results = []

    alice_cls, bob_cls = {
        "teleportation": (AliceTeleportation, BobTeleportation),
        "pingpong": (AlicePingpongTeleportation, BobPingpongTeleportation),
    }.get(experiment, (AliceProgram, BobProgram))

    for comb in combinations:
        for param, value in zip(sweep_params, comb):
            if hasattr(cfg.links[0], "cfg") and cfg.links[0].cfg is not None:
                cfg.links[0].cfg[param] = value
            else:
                cfg.stacks[0].qdevice_cfg[param] = value

        _, bob_results = run(
            config=cfg,
            programs={
                "Alice": alice_cls(num_epr_rounds=epr_rounds),
                "Bob": bob_cls(num_epr_rounds=epr_rounds),
            },
            num_times=num_experiments,
        )
        avg_fid = np.mean([res[0] for res in bob_results]) * 100
        avg_time = np.mean([res[1] for res in bob_results])
        results.append(
            dict(zip(sweep_params, comb), **{FIDELITY: avg_fid, TIME: avg_time})
        )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f"{experiment}_results.csv"), index=False)
    return df


def main():
    """Parses command-line arguments, runs simulations, and generates plots.

    This function reads configuration arguments, executes parameter sweeps
    if specified, and generates corresponding plots for visualization.

    """
    parser = argparse.ArgumentParser(
        description="Run simulation, perform parameter sweep, and generate plots."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="cnot",
        help="Experiment to simulate (cnot, teleportation, pingpong).",
    )
    parser.add_argument(
        "--epr_rounds", type=int, default=10, help="Number of EPR rounds (default 10)."
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=10,
        help="Number of experiments per combination (default 10).",
    )
    parser.add_argument(
        "--sweep_params",
        type=str,
        help="Comma-separated list of configuration parameter names to sweep (e.g., 'T1,T2').",
    )
    parser.add_argument(
        "--ranges",
        nargs="+",
        type=str,
        help="One range string per parameter (format: 'start,end,points').",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results."
    )
    args = parser.parse_args()

    unique_output_dir = create_unique_dir(args.output_dir)
    print(f"Using output directory: {unique_output_dir}")

    cfg = StackNetworkConfig.from_file(args.config)

    if args.sweep_params and args.ranges:
        # Ensure the number of parameters matches the number of ranges
        sweep_params = [p.strip() for p in args.sweep_params.split(",")]
        if len(sweep_params) != len(args.ranges):
            raise ValueError(
                "The number of sweep parameters must match the number of range strings provided."
            )

        # Run parameter sweep and generate results
        df = sweep_parameters(
            cfg,
            args.epr_rounds,
            args.num_experiments,
            sweep_params,
            args.ranges,
            args.experiment,
            output_dir=unique_output_dir,
        )
        print("Sweep completed. Preview of results:")
        print(df.head())

        # Generate correlation heatmap
        plot_parameter_metric_correlation(
            df, sweep_params, [FIDELITY, TIME], unique_output_dir, args.experiment
        )

        # Build parameter range dictionary
        param_range_dict = {
            param: parse_range(rng_str)
            for param, rng_str in zip(sweep_params, args.ranges)
        }

        # Generate visualizations
        plot_combined_3d_surfaces(
            df, sweep_params, param_range_dict, unique_output_dir, args.experiment
        )
        plot_combined_heatmaps(
            df, sweep_params, param_range_dict, unique_output_dir, args.experiment
        )
    else:
        print("No sweep parameters provided. Running single configuration simulation.")


if __name__ == "__main__":
    main()
