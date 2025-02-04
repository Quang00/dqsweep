import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run

from application import AliceProgram, BobProgram
from teleportation import AliceTeleportation, BobTeleportation


def main():
    parser = argparse.ArgumentParser(
        description="Distributed CNOT gate simulation and analyze quantum network parameter effects."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration of the quantum network file.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cnot",
        help="The method to perfom the nonlocal cnot (cnot, teleportation).",
    )
    parser.add_argument(
        "--epr_rounds",
        type=int,
        default=10,
        help="Number of EPR rounds. Default is 10.",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=10,
        help="Number of experiments. Default is 10.",
    )
    parser.add_argument(
        "--plot_parameter_effects",
        nargs=2,
        metavar=("PARAM1", "PARAM2"),
        help="Two parameters to analyze together.",
    )
    parser.add_argument(
        "--param1_range",
        type=str,
        help="Range for the first parameter: 'start,end,points'.",
    )
    parser.add_argument(
        "--param2_range",
        type=str,
        help="Range for the second parameter: 'start,end,points'.",
    )

    args = parser.parse_args()

    # Load configuration
    cfg = StackNetworkConfig.from_file(args.config)

    # Parse ranges
    param1_range = (
        parse_range(args.plot_parameter_effects, args.param1_range)
        if args.param1_range
        else np.linspace(0.0, 0.8, 10)
    )
    param2_range = (
        parse_range(args.plot_parameter_effects, args.param2_range)
        if args.param2_range
        else np.linspace(0.0, 0.8, 10)
    )

    # Analyze parameter effects for two parameters or calculate average fidelity
    if args.plot_parameter_effects:
        analyze_two_parameters(
            cfg,
            args.epr_rounds,
            args.num_experiments,
            args.plot_parameter_effects,
            param1_range,
            param2_range,
        )
    else:
        calculate_average_fidelity(cfg, args.method, args.epr_rounds, args.num_experiments)


def parse_range(params, range_str):
    """Parse a range string in the format 'start,end,points'."""
    if not range_str or not params:
        return None
    try:
        param1, param2 = params
        log_scale_params = ['T1', 'T2', 'length']
        start, end, points = map(float, range_str.split(","))
        if param1 in log_scale_params or param2 in log_scale_params:
            return np.logspace(start, end, int(points))
        else:
            return np.linspace(start, end, int(points))
    except ValueError:
        print("Ranges must be in the format 'start,end,points', with numeric values.")


def analyze_two_parameters(
    cfg,
    epr_rounds,
    num_experiments,
    params_to_plot,
    param1_values,
    param2_values,
    output_dir="results",
):
    """Analyze and save a heatmap of the combined effect of two parameters on fidelity."""
    os.makedirs(output_dir, exist_ok=True)

    param1, param2 = params_to_plot
    depolarise_link_param = ['fidelity', 'prob_succcess']

    # Initialize fidelity matrix
    fidelity_matrix = np.zeros((len(param1_values), len(param2_values)))

    # Iterate through parameter combinations
    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            try:
                if param1 in depolarise_link_param or param2 in depolarise_link_param:
                    cfg.links[0].cfg[param1] = val1
                    cfg.links[0].cfg[param2] = val2
                else:
                    cfg.stacks[0].qdevice_cfg[param1] = val1
                    cfg.stacks[0].qdevice_cfg[param2] = val2

                _, results = run(
                    config=cfg,
                    programs={
                        "Alice": AliceProgram(num_epr_rounds=epr_rounds),
                        "Bob": BobProgram(num_epr_rounds=epr_rounds),
                    },
                    num_times=num_experiments,
                )
                results = [results[i][0] for i in range(len(results))]
                fidelity_matrix[i, j] = np.average(results) * 100
            except ValueError as e:
                print(f"Skipping {param1}={val1}, {param2}={val2}: {e}")
                fidelity_matrix[i, j] = 0
    plot_2d_heatmap(
        param1, param2, param1_values, param2_values, fidelity_matrix, output_dir
    )


def calculate_average_fidelity(cfg, method, epr_rounds, num_experiments):
    """Calculate and print the average fidelity over multiple experiments."""
    try:
        if method == "teleportation":
            alice_method = AliceTeleportation(num_epr_rounds=epr_rounds)
            bob_method = BobTeleportation(num_epr_rounds=epr_rounds)
        else:
            alice_method = AliceProgram(num_epr_rounds=epr_rounds)
            bob_method = BobProgram(num_epr_rounds=epr_rounds)
        _, results = run(
            config=cfg,
            programs={
                "Alice": alice_method,
                "Bob": bob_method,
            },
            num_times=num_experiments,
        )
        print(f"Method used: {method}")
        fidelities = [results[i][0] for i in range(len(results))]
        simulation_times = [results[i][1] for i in range(len(results))]
        avg_fidelity = np.average(fidelities) * 100
        avg_simulation_times = np.average(simulation_times)
        print(
            f"Average fidelity over {num_experiments} experiments: {avg_fidelity:.2f}%"
        )
        print(
            f"Average simulation time over {num_experiments} experiments: {avg_simulation_times:.2f} ms"
        )
    except ValueError as e:
        print(f"Error during simulation: {e}")


def plot_2d_heatmap(
    param1, param2, param1_values, param2_values, fidelity_matrix, output_dir
):
    """Plot the 2D heatmap for two parameters."""
    plt.figure()
    plt.imshow(
        fidelity_matrix,
        extent=[
            param2_values[0],
            param2_values[-1],
            param1_values[0],
            param1_values[-1],
        ],
        origin="lower",
        aspect="auto",
        cmap="coolwarm"
    )
    plt.colorbar(label="Fidelity (%)")
    plt.title(
        f"Effect of {param1.replace('_', ' ').capitalize()} and {param2.replace('_', ' ').capitalize()} on Fidelity"
    )
    plt.xlabel(f"{param2.replace('_', ' ').capitalize()}")
    plt.ylabel(f"{param1.replace('_', ' ').capitalize()}")
    filename = os.path.join(output_dir, f"heatmap_{param1}_{param2}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved heatmap for {param1} and {param2} to {filename}")


if __name__ == "__main__":
    main()
