import argparse
import os
import logging

import matplotlib.pyplot as plt
import numpy as np
from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run

from nonlocal_cnot import AliceProgram, BobProgram
from teleportation import AliceTeleportation, BobTeleportation
from pingpong import AlicePingpongTeleportation, BobPingpongTeleporation


def main():
    parser = argparse.ArgumentParser(
        description="Analysis of fidelity/latency of multiple parameters of a quantum network for different experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration of the quantum network file.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="cnot",
        help="The experiment to simulate (cnot, teleportation, pingpong).",
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

    if args.experiment == "pingpong":
        cfg.stacks[0].qdevice_cfg['num_qubits'] = args.num_experiments * args.epr_rounds

    # Analyze parameter effects for two parameters or calculate average fidelity
    if args.plot_parameter_effects:
        analyze_two_parameters(
            cfg,
            args.epr_rounds,
            args.num_experiments,
            args.plot_parameter_effects,
            param1_range,
            param2_range,
            args.experiment
        )
    else:
        calculate_average_fidelity(cfg, args.experiment, args.epr_rounds, args.num_experiments)


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
    experiment,
    output_dir="results",
):
    """
    Analyze and save heatmaps of the combined effect of two parameters on fidelity and simulation time.
    For each combination of parameter values, we run the simulation to obtain two metrics:
      - Fidelity (extracted as the first element in each result, multiplied by 100)
      - Simulation Time (extracted as the second element in each result)
    After averaging over experiments, two heatmaps are generated.
    """
    os.makedirs(output_dir, exist_ok=True)

    param1, param2 = params_to_plot
    depolarise_link_param = ['fidelity', 'prob_succcess']

    # Initialize matrices for both fidelity and simulation time
    fidelity_matrix = np.zeros((len(param1_values), len(param2_values)))
    simulation_time_matrix = np.zeros((len(param1_values), len(param2_values)))

    if  experiment == "teleportation":
            alice_method = AliceTeleportation(num_epr_rounds=epr_rounds)
            bob_method = BobTeleportation(num_epr_rounds=epr_rounds)
    elif experiment == "pingpong":
            alice_method = AlicePingpongTeleportation(num_epr_rounds=epr_rounds)
            bob_method = BobPingpongTeleporation(num_epr_rounds=epr_rounds)
            #alice_method.logger.setLevel(logging.INFO)
            #bob_method.logger.setLevel(logging.INFO)
    else:
            alice_method = AliceProgram(num_epr_rounds=epr_rounds)
            bob_method = BobProgram(num_epr_rounds=epr_rounds)

    # Iterate through parameter combinations
    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            try:
                # Update configuration based on the parameter location
                if param1 in depolarise_link_param or param2 in depolarise_link_param:
                    cfg.links[0].cfg[param1] = val1
                    cfg.links[0].cfg[param2] = val2
                else:
                    cfg.stacks[0].qdevice_cfg[param1] = val1
                    cfg.stacks[0].qdevice_cfg[param2] = val2

                # Run the simulation
                alice_results, bob_results = run(config=cfg, programs={"Alice": alice_method, "Bob": bob_method}, num_times=num_experiments)

                fidelities = [bob_results[k][0] for k in range(len(bob_results))]
                simulation_times = [bob_results[k][1] for k in range(len(bob_results))]

                if len(fidelities) > 0 and len(simulation_times) > 0:
                    fidelity_matrix[i, j] = np.average(fidelities) * 100
                    simulation_time_matrix[i, j] = np.average(simulation_times)
            except ValueError as e:
                print(f"Skipping {param1}={val1}, {param2}={val2}: {e}")
                fidelity_matrix[i, j] = 0
                simulation_time_matrix[i, j] = 0

    plot_2d_heatmap(
        param1, param2, param1_values, param2_values,
        fidelity_matrix, output_dir, experiment, metric="Fidelity (%)"
    )
    plot_2d_heatmap(
        param1, param2, param1_values, param2_values,
        simulation_time_matrix, output_dir, experiment, metric="Simulation Time (ms)"
    )


def calculate_average_fidelity(cfg, experiment, epr_rounds, num_experiments):
    """Calculate and print the average fidelity and simulation time over multiple experiments."""
    try:
        if experiment == "teleportation":
            alice_method = AliceTeleportation(num_epr_rounds=epr_rounds)
            bob_method = BobTeleportation(num_epr_rounds=epr_rounds)
        elif experiment == "pingpong":
            alice_method = AlicePingpongTeleportation(num_epr_rounds=epr_rounds)
            bob_method = BobPingpongTeleporation(num_epr_rounds=epr_rounds)
            alice_method.logger.setLevel(logging.INFO)
            bob_method.logger.setLevel(logging.INFO)
        else:
            alice_method = AliceProgram(num_epr_rounds=epr_rounds)
            bob_method = BobProgram(num_epr_rounds=epr_rounds)

        alice_results, bob_results = run(config=cfg, programs={"Alice": alice_method, "Bob": bob_method,}, num_times=num_experiments)

        if experiment == "pingpong":
            alice_fidelities = [alice_results[i][0] for i in range(len(alice_results))]
            alice_simulation_times = [alice_results[i][1] for i in range(len(alice_results))]
            bob_fidelities = [bob_results[i][0] for i in range(len(bob_results))]
            bob_simulation_times = [bob_results[i][1] for i in range(len(bob_results))]

            if len(alice_fidelities) > 0 and len(bob_fidelities) > 0 and len(alice_simulation_times) > 0 and len(bob_simulation_times) > 0:
                avg_alice_fidelity = np.average(alice_fidelities) * 100
                avg_alice_simulation_times = np.average(alice_simulation_times)
                avg_bob_fidelity = np.average(bob_fidelities) * 100
                avg_bob_simulation_times = np.average(bob_simulation_times)
            
                print(f"Average fidelity for Alice over {num_experiments} experiments: {avg_alice_fidelity:.2f}%")
                print(f"Average fidelity for Bob over {num_experiments} experiments: {avg_bob_fidelity:.2f}%")
                print(f"Average simulation time for Alice over {num_experiments} experiments: {avg_alice_simulation_times:.2f} ms")
                print(f"Average simulation time for Bob over {num_experiments} experiments: {avg_bob_simulation_times:.2f} ms")
            else:
                print("The experiment fails. Make sure to provide a valid configuration.")
        else:
            fidelities = [bob_results[i][0] for i in range(len(bob_results))]
            simulation_times = [bob_results[i][1] for i in range(len(bob_results))]
            
            if len(fidelities) > 0 and len(simulation_times) > 0:
                avg_fidelity = np.average(fidelities) * 100
                avg_simulation_times = np.average(simulation_times)

                print(f"Average fidelity over {num_experiments} experiments: {avg_fidelity:.2f}%")
                print(f"Average simulation time over {num_experiments} experiments: {avg_simulation_times:.2f} ms")
            else:
                print("The experiment fails. Make sure to provide a valid configuration.")
    except ValueError as e:
        print(f"Error during simulation: {e}")


def plot_2d_heatmap(
    param1, param2, param1_values, param2_values, metric_matrix, output_dir, experiment, metric="Fidelity (%)"
):
    """
    Plot a 2D heatmap for two parameters.

    The heatmap displays the provided metric (e.g. 'Fidelity (%)' or 'Simulation Time (ms)').
    """

    cmap_style = "magma" if metric == "Fidelity (%)" else "viridis"

    plt.figure(figsize=(10,8))
    plt.imshow(
        metric_matrix,
        extent=[
            param2_values[0],
            param2_values[-1],
            param1_values[0],
            param1_values[-1],
        ],
        origin="lower",
        aspect="auto",
        cmap=cmap_style
    )
    plt.colorbar(label=metric)

    metric_name = metric.split(" (")[0] if "(" in metric else metric

    param1 = " ".join(param1.split('_')[:3])
    param2 = " ".join(param2.split('_')[:3])

    plt.title(
        f"{experiment.capitalize()}: Effect of {param1.replace('_', ' ').capitalize()} and {param2.replace('_', ' ').capitalize()} on {metric_name}"
    )
    plt.xlabel(f"{param2.replace('_', ' ').capitalize()}")
    plt.ylabel(f"{param1.replace('_', ' ').capitalize()}")
    filename = os.path.join(output_dir, f"{experiment}_heatmap_{param1}_{param2}_{metric_name.replace(' ', '_').lower()}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved heatmap for {param1} and {param2} on {metric} to {filename}")


if __name__ == "__main__":
    main()
