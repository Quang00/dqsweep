import itertools
import os
from typing import Generator, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netqasm.sdk import Qubit
from netsquid.qubits.dmutil import dm_fidelity

from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from squidasm.sim.stack.program import ProgramContext
from squidasm.util import get_qubit_state
from squidasm.util.routines import teleport_recv, teleport_send

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
def truncate_param(name: str, char: str = "_", n: int = 4) -> str:
    """Truncates a parameter name to improve readability in plots.

    Args:
        name (str): Parameter name to truncate.
        char (str, optional): Character to split the name. Defaults to "_".
        n (int, optional): Number of tokens to keep. Defaults to 4.

    Returns:
        str: Truncated parameter name.
    """
    return " ".join(name.split(char)[:n]).capitalize()


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
            return np.logspace(start, end, int(points))
        return np.linspace(start, end, int(points))
    except ValueError:
        raise ValueError("Invalid format. Use 'start,end,points'.") from None


def compute_fidelity(qubit: Qubit, owner: str, dm_state: np.ndarray) -> float:
    """
    Computes the fidelity between the density matrix of a given qubit and
    the density matrix constructed from a reference state vector.

    Args:
        qubit (Qubit): The qubit whose state is to be evaluated.
        owner (str): The owner of the qubit.
        dm_state (np.ndarray): The reference state vector.

    Returns:
        float: The fidelity between the two density matrices.
    """
    dm = get_qubit_state(qubit, owner, full_state=True)
    dm_ref = np.outer(dm_state, np.conjugate(dm_state))
    fidelity = dm_fidelity(dm, dm_ref, dm_check=False)

    return fidelity


def metric_correlation(
    df: pd.DataFrame,
    sweep_params: list,
    metric_cols: list,
    output_dir: str,
    experiment: str,
):
    """Computes and generate a txt file for parameter-performance correlation.

    Args:
        df (pd.DataFrame): Dataframe containing parameter values and metrics.
        sweep_params (list): List of parameters being swept.
        metric_cols (list): List of performance metrics.
        output_dir (str): Directory to save the plot.
        experiment (str): Experiment name.
    """
    # Calculate the correlation for the selected columns
    corr = df[sweep_params + metric_cols].corr().loc[sweep_params, metric_cols]
    filename = os.path.join(output_dir, f"{experiment}_corr.txt")

    with open(filename, "w") as f:
        f.write("Parameter\t" + "\t".join(metric_cols) + "\n")
        for param in sweep_params:
            row = [f"{corr.loc[param, metric]:.3f}" for metric in metric_cols]
            f.write(f"{param}\t" + "\t".join(row) + "\n")

    print(f"Saved correlation values to {filename}")


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_heatmap(ax, df, p, q, metric, params, exp, epr_rounds):
    """
    Plot an individual heatmap on a given axes.

    Args:
        ax: Matplotlib Axes object.
        df (pd.DataFrame): Dataframe containing experiment results.
        p (str): Name of the parameter for the y-axis.
        q (str): Name of the parameter for the x-axis.
        metric (dict): Dictionary with keys 'name', 'cmap', and 'file_label'.
        params (dict): Dictionary mapping parameters to their ranges.
        exp (str): Name of the experiment.
        epr_rounds (int): Number of epr rounds.
    """
    pivot = df.pivot_table(index=p, columns=q, values=metric["name"])
    metric_matrix = pivot.values

    im = ax.imshow(
        metric_matrix,
        extent=[
            params[q][0],
            params[q][-1],  # X-axis
            params[p][0],
            params[p][-1],  # Y-axis
        ],
        origin="lower",
        aspect="auto",
        cmap=metric["cmap"],
    )

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(metric["name"], fontsize=14)

    var1 = truncate_param(q)
    var2 = truncate_param(p)

    ax.set_xlabel(var1, fontsize=14)
    ax.set_ylabel(var2, fontsize=14)

    title_suffix = f" with {epr_rounds} hops" if exp == "pingpong" else ""
    ax.set_title(
        f"{exp.capitalize()}: {var2} vs {var1}{title_suffix}",
        fontsize=16,
        fontweight="bold",
    )


def plot_combined_heatmaps(
    df: pd.DataFrame,
    sweep_params: list,
    params: dict,
    output_dir: str,
    exp: str,
    epr_rounds: int,
    separate_files: bool = False,
):
    """
    Generates heatmaps from experiment results.

    If `separate_files` is True, two separate figures will be created:
    one for Average Fidelity and one for Average Simulation Time.
    Otherwise, a single combined figure is generated.

    Args:
        df (pd.DataFrame): Dataframe containing experiment results.
        sweep_params (list): List of swept parameters.
        params (dict): Dictionary mapping parameters to their ranges.
        output_dir (str): Directory to save the generated figures.
        exp (str): Name of the experiment.
        epr_rounds (int): Number of epr rounds.
        separate_files (bool, optional): If True, save in a separate file.
    """
    pairs = list(itertools.combinations(sweep_params, 2))
    metrics = [
        {
            "name": "Average Fidelity (%)",
            "cmap": "magma",
            "file_label": "fidelity",
        },
        {
            "name": "Average Simulation Time (ms)",
            "cmap": "viridis",
            "file_label": "sim_times",
        },
    ]

    if separate_files:
        # Generate and save one figure per metric.
        figsize = (12 * len(pairs), 8)
        for metric in metrics:
            fig, axes = plt.subplots(1, len(pairs), figsize=figsize)
            if len(pairs) == 1:
                axes = [axes]
            for ax, (p, q) in zip(axes, pairs):
                plot_heatmap(ax, df, p, q, metric, params, exp, epr_rounds)
            plt.tight_layout()
            suffix = f"{(epr_rounds + 1) // 2}" if exp == "pingpong" else ""
            filename = os.path.join(
                output_dir, f"{exp}_heat_{metric['file_label']}_{suffix}.png"
            )
            plt.savefig(filename, dpi=300)
            plt.close(fig)
            print(f"Saved {metric['name']} heatmap to {filename}")
    else:
        # Create a single combined figure with one row per metric.
        figsize = (12 * len(pairs), 8 * len(metrics))
        fig, axes = plt.subplots(len(metrics), len(pairs), figsize=figsize)
        # Ensure axes is 2D even when there's only one pair.
        if len(pairs) == 1:
            axes = np.array([axes]).reshape(len(metrics), 1)
        for i, metric in enumerate(metrics):
            for j, (p, q) in enumerate(pairs):
                plot_heatmap(axes[i, j], df, p, q, metric, params, exp, epr_rounds)
        plt.tight_layout()
        filename = os.path.join(output_dir, f"{exp}_heatmaps.png")
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"Saved combined heatmaps to {filename}")


def run_simulation(
    config: str,
    epr_rounds: int = 10,
    num_times: int = 10,
    alice_cls=None,
    bob_cls=None,
):
    """
    Runs a simulation with the given configuration and program classes.

    Args:
        config (str): Path to the network configuration YAML file.
        epr_rounds (int): Number of EPR rounds to execute in the simulation.
        num_times (int): Number of simulation repetitions.
        alice_cls (type): The class implementing Alice's program.
        bob_cls (type): The class implementing Bob's program.

    Returns:
        dict: A dictionary containing simulation results for each node.
    """
    if alice_cls is None or bob_cls is None:
        raise ValueError("Alice class and Bob class must be provided.")

    # Load the network configuration.
    cfg = StackNetworkConfig.from_file(config)

    alice_program = alice_cls(num_epr_rounds=epr_rounds)
    bob_program = bob_cls(num_epr_rounds=epr_rounds)

    # Run the simulation with the provided configuration.
    results = run(
        config=cfg,
        programs={"Alice": alice_program, "Bob": bob_program},
        num_times=num_times,
    )

    return results


# =============================================================================
# Routines
# =============================================================================
def pingpong_initiator(
    qubit: Qubit, context: ProgramContext, peer_name: str, num_rounds: int = 3
):
    """
    Executes the ping‐pong teleportation protocol for the initiator.

    In even rounds, the provided qubit is sent to the peer.
    In odd rounds, the initiator receives the qubit.
    The formal return is a generator and requires use of `yield from`
    in usage in order to function as intended.

    Args:
        qubit (Qubit): The qubit to be teleported.
        context (ProgramContext): Connection, csockets, and epr_sockets.
        peer_name (str): Name of the peer.
        num_rounds (int): Number of ping‐pong rounds.
    """
    if num_rounds % 2 == 0:
        raise ValueError("It must be odd for a complete ping-pong exchange.")

    for round_num in range(num_rounds):
        if round_num % 2 == 0:
            # Even round: send the qubit.
            yield from teleport_send(qubit, context, peer_name)
        else:
            # Odd round: receive a new qubit from the peer.
            qubit = yield from teleport_recv(context, peer_name)

    yield from context.connection.flush()


def pingpong_responder(
    context: ProgramContext, peer_name: str, num_rounds: int = 3
) -> Generator[None, None, Qubit]:
    """
    Executes the complementary ping‐pong teleportation protocol
    for the responder.

    The responder starts without a qubit and in the first (even) round
    receives one. In odd rounds he sends the qubit. After completing
    the rounds, Bob returns the final qubit he holds.
    The formal return is a generator and requires use of `yield from`
    in usage in order to function as intended.

    Args:
        context (ProgramContext): Connection, csockets, and epr_sockets.
        peer_name (str): Name of the peer.
        num_rounds (int): Number of ping‐pong rounds.

    Returns:
        Generator[None, None, Qubit]: The final teleported qubit.
    """
    if num_rounds % 2 == 0:
        raise ValueError("It must be odd for a complete ping-pong exchange.")

    qubit = None

    for round_num in range(num_rounds):
        if round_num % 2 == 0:
            # Even round: receive a qubit from the peer.
            qubit = yield from teleport_recv(context, peer_name)
        else:
            # Odd round: send the qubit to the peer.
            yield from teleport_send(qubit, context, peer_name)

    yield from context.connection.flush()

    return qubit
