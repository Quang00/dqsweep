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

import numpy as np
import pandas as pd

from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run

from nonlocal_cnot import AliceProgram, BobProgram
from pingpong import AlicePingpongTeleportation, BobPingpongTeleportation
from teleportation import AliceTeleportation, BobTeleportation
from utils import (
    create_subdir,
    parse_range,
    plot_combined_3d_surfaces,
    plot_combined_heatmaps,
    plot_parameter_metric_correlation,
)


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
        param: parse_range(rng_str, param)
        for param, rng_str in zip(sweep_params, ranges)
    }

    combinations = list(
        itertools.product(*[param_ranges[name] for name in sweep_params])
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
        all_fid_results = [res[0] for res in bob_results]
        all_time_results = [res[1] for res in bob_results]

        avg_fid = np.mean(all_fid_results) * 100
        avg_time = np.mean(all_time_results)

        results.append(
            dict(
                zip(sweep_params, comb),
                **{
                    "Fidelity Results": all_fid_results,
                    "Simulation Time Results": all_time_results,
                    "Average Fidelity (%)": avg_fid,
                    "Average Simulation Time (ms)": avg_time,
                },
            )
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
        description="Simulate quantum network experiments with sweep parameters."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="cnot",
        help="Experiment to simulate (cnot, teleportation, pingpong).",
    )
    parser.add_argument(
        "--epr_rounds", type=int, default=10, help="Number of EPR rounds."
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

    if args.experiment == "pingpong" and args.epr_rounds % 2 == 0:
        print("ArgumentError: You need to specify an odd number of epr rounds.")
        return

    unique_output_dir = create_subdir(args.output_dir, args.experiment, args.sweep_params)
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
            df,
            sweep_params,
            ["Average Fidelity (%)", "Average Simulation Time (ms)"],
            unique_output_dir,
            args.experiment,
        )

        # Build parameter range dictionary
        param_range_dict = {
            param: parse_range(rng_str, param)
            for param, rng_str in zip(sweep_params, args.ranges)
        }

        # Generate visualizations
        plot_combined_3d_surfaces(
            df, sweep_params, param_range_dict, unique_output_dir, args.experiment, args.epr_rounds
        )
        plot_combined_heatmaps(
            df, sweep_params, param_range_dict, unique_output_dir, args.experiment, args.epr_rounds, separate_files=False
        )
    else:
        print("No sweep parameters provided. Running single configuration simulation.")


if __name__ == "__main__":
    main()
