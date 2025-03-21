"""
Dqsweep
=======================================================

Overview:
---------
This script simulates distributed quantum experiments using a configurable
setup. It allows to sweep one or more parameters over defined ranges and then
execute multiple simulation runs in parallel for each parameter combination.
For every run, it calculates two performance metrics: the average fidelity
(as a percentage) and the average simulation time (in milliseconds).

Process:
--------
1. Parameter Sweep:
   - Generates all combinations of values for the specified parameters.
   - For each combination, it executes the simulation a certain amount of times
    which is configurable by the user and computes the mean fidelity and
    simulation time.

2. Data Generation:
   - Raw results and computed averages for each parameter combination are
    stored in a pandas DataFrame and saved as a CSV file.
   - Parameter–Performance Correlation txt file is provided and shows the
    correlations between the input parameters and the performance metrics.

3. Visualization:
   - 2D Heatmaps: These are generated for each performance metric based on
    each unique pair of swept parameters.

Usage:
------
Run the script with the following command-line arguments:

  --config         Path to the YAML config file that defines the network setup
  --experiment     Experiment to simulate (e.g., cnot, pingpong, dgrover2, ...)
  --epr_rounds     Number of EPR rounds per simulation
  --num_experiments Number of simulation runs per parameter combination
  --sweep_params   Comma-separated list of parameter names for the sweep.
  --ranges         For each swept parameter, provide a range in the format
                   "start,end,points", where start is the initial value, end
                   is the final value, and points is the number of values
                   (or steps) to generate between start and end.
  --output_dir     Directory where the CSV file, txt file and generated plots
                   will be saved.
"""

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd

from experiments.dgrover import GroverControl, GroverTarget
from experiments.dgrover_2 import AliceDGrover2, BobDGrover2
from experiments.dqft_2 import AliceDQFT2, BobDQFT2
from experiments.nonlocal_cnot_teledata import (
    Alice2Teleportations,
    Bob2Teleportations,
)
from experiments.nonlocal_cnot_telegate import AliceProgram, BobProgram
from experiments.nonlocal_toffoli import (
    AliceToffoli,
    BobToffoli,
    CharlieToffoli
)
from experiments.pingpong import (
    AlicePingpongTeleportation,
    BobPingpongTeleportation
)
from squidasm.run.stack.config import StackNetworkConfig
from utils.helper import (
    check_sweep_params_input,
    create_subdir,
    metric_correlation,
    parallelize_comb,
    parse_range,
)
from utils.plots import plot_combined_heatmaps


# =============================================================================
# Parameter Sweep Function
# =============================================================================
def sweep_parameters(
    cfg: StackNetworkConfig,
    rounds: int,
    num_experiments: int,
    sweep_params: list,
    ranges: list,
    experiment: str,
    output_dir: str,
) -> pd.DataFrame:
    """Performs a parameter sweep, runs simulations, and stores results.

    Args:
        cfg (StackNetworkConfig): Network configuration.
        rounds (int): Number of EPR rounds.
        num_experiments (int): Number of experiments per configuration.
        sweep_params (list): Parameters to sweep.
        ranges (list): Ranges for each parameter.
        experiment (str): Experiment name.
        output_dir (str): Directory to save results.

    Returns:
        pd.DataFrame: DataFrame of results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parse parameter ranges.
    param_ranges = {
        param: parse_range(rng_str, param)
        for param, rng_str in zip(sweep_params, ranges, strict=False)
    }
    # Create all combinations of parameters.
    comb_list = list(
        itertools.product(*(param_ranges[param] for param in sweep_params))
    )

    # Map experiments to their program classes.
    experiment_map = {
        "cnot_teledata": (Alice2Teleportations, Bob2Teleportations),
        "dgrover": (GroverControl, GroverTarget),
        "dgrover2": (AliceDGrover2, BobDGrover2),
        "dqft2": (AliceDQFT2, BobDQFT2),
        "pingpong": (AlicePingpongTeleportation, BobPingpongTeleportation),
        "toffoli": (AliceToffoli, BobToffoli, CharlieToffoli),
    }
    classes = experiment_map.get(experiment, (AliceProgram, BobProgram))
    names = ["Alice", "Bob"] + (["Charlie"] if len(classes) > 2 else [])

    if experiment == "dgrover":
        peers = names[:-1]
        trgt_peer = names[-1]
        programs = {name: GroverControl(trgt_peer, rounds) for name in peers}
        programs[trgt_peer] = GroverTarget(peers, rounds)
    else:
        programs = {
            name: cls(num_epr_rounds=rounds)
            for name, cls in zip(names, classes, strict=False)
        }

    func = partial(
        parallelize_comb,
        cfg=cfg,
        sweep_params=sweep_params,
        num_experiments=num_experiments,
        programs=programs,
    )

    # Use ProcessPoolExecutor to parallelize the combinations.
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(func, comb_list))

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"{experiment}_results.csv")
    df.to_csv(csv_path, index=False)
    return df


def main():
    """Main entry point for the Distributed Quantum Experiments Script.

    This function launches the simulation, starting from the parsing
    command-line arguments to generating performance plots (Heat maps),
    raw results (csv file), and correlation analysis (txt file).

    """
    parser = argparse.ArgumentParser(
        description="Simulate distributed quantum experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configurations/perfect.yaml",
        help="Path to the configuration.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="cnot",
        help="Distributed experiments (e.g., cnot, pingpong, dgrover2, ...).",
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
        default="single_qubit_gate_depolar_prob,two_qubit_gate_depolar_prob",
        help="Comma-separated list of configuration parameter names to sweep",
    )
    parser.add_argument(
        "--ranges",
        nargs="+",
        type=str,
        default=["0.0,0.8,10", "0.0,0.8,10"],
        help="One range string per parameter (format: 'start,end,points').",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )
    args = parser.parse_args()

    output_dir = create_subdir(
        args.output_dir,
        args.experiment,
        args.sweep_params
    )
    print(f"Using output directory: {output_dir}")

    cfg = StackNetworkConfig.from_file(args.config)

    check_sweep_params_input(cfg, args.sweep_params)

    if args.sweep_params and args.ranges:
        # Ensure the number of parameters matches the number of ranges
        sweep_params = [p.strip() for p in args.sweep_params.split(",")]
        if len(sweep_params) != len(args.ranges):
            raise ValueError(
                "Number of sweep parameters must match number of range strings"
            )

        # Run parameter sweep and generate results
        df = sweep_parameters(
            cfg,
            args.epr_rounds,
            args.num_experiments,
            sweep_params,
            args.ranges,
            args.experiment,
            output_dir=output_dir,
        )
        print("Sweep completed. Preview of results:")
        print(df.head())

        # Generate a txt file with the correlation values
        metric_correlation(
            df,
            sweep_params,
            ["Average Fidelity (%)", "Average Simulation Time (ms)"],
            output_dir,
            args.experiment,
        )

        # Build parameter range dictionary
        param_range_dict = {
            param: parse_range(rng_str, param)
            for param, rng_str in zip(sweep_params, args.ranges, strict=False)
        }

        # Generate heat maps for each metrics or a combined heat map
        plot_combined_heatmaps(
            df,
            sweep_params,
            param_range_dict,
            output_dir,
            args.experiment,
            args.epr_rounds,
            separate_files=True,
        )


if __name__ == "__main__":
    main()
