import copy
import os
from typing import Union

import numpy as np
import pandas as pd
from netqasm.sdk import Qubit
from netsquid.qubits.dmutil import dm_fidelity

from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from squidasm.util import get_qubit_state


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


def compute_fidelity(
    qubit: Qubit, owner: str, dm_state: np.ndarray, full_state: bool = True
) -> float:
    """Computes the fidelity between the density matrix of a given qubit and
    the density matrix constructed from a reference state vector.

    Args:
        qubit (Qubit): The qubit whose state is to be evaluated.
        owner (str): The owner of the qubit.
        dm_state (np.ndarray): The reference state vector.
        full_state (bool): Flag to retrieve the full entangled state.

    Returns:
        float: The fidelity between the two density matrices.
    """
    dm = get_qubit_state(qubit, owner, full_state=full_state)
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


def simulate_and_compute_avg_fidelity(
    config: StackNetworkConfig,
    programs: dict = None,
) -> float:
    """Runs a simulation with the given configuration and program classes and
    compute the average fidelilty of the simulation.

    Args:
        config (str): Path to the network configuration YAML file.
        programs (dict): A dictionary mapping program names to their classes.

    Returns:
        float: Average fidelity of the simulation.
    """
    if programs is None or not programs:
        raise ValueError("At least one class must be provided.")

    # Run the simulation with the provided configuration.
    all_results = run(
        config=config,
        programs=programs,
        num_times=10,
    )

    results = all_results[len(programs) - 1]

    # Compute the average fidelity.
    all_fid_results = [res[0] for res in results]
    avg_fidelity = np.mean(all_fid_results)

    return avg_fidelity


def check_sweep_params_input(cfg: StackNetworkConfig, params: str) -> bool:
    """Check that all sweep parameters are found in the configuration.

    Args:
        cfg (StackNetworkConfig): Network configuration.
        params (str): Parameters to sweep.

    Returns:
        bool: True if all sweep parameters are found.

    Raises:
        ValueError: If one or more sweep parameters are missing in the config.
    """
    sweep_set = {param.strip() for param in params.split(",")}
    cfg_set = {token.strip("',:") for token in str(cfg).split()}

    missing = sweep_set - cfg_set
    if missing:
        raise ValueError("Please provide parameters that exist in the config.")
    return True


def update_cfg(cfg: StackNetworkConfig, map_param: dict) -> None:
    """Update the configuration for every link and stack with the
    given parameter values.

    Args:
        cfg (StackNetworkConfig): Network configuration.
        map_param (dict): Map parameter names to their values.
    """
    for param, value in map_param.items():
        for link in cfg.links:
            if getattr(link, "link_cfg", None) is not None:
                link.cfg[param] = value
        for stack in cfg.stacks:
            if getattr(stack, "qdevice_cfg", None) is not None:
                stack.qdevice_cfg[param] = value


def parallelize_comb(comb, cfg, sweep_params, num_experiments, programs):
    """Parallelize a single combination of parameters

    Args:
        comb (tuple): One combination of parameters value.
        cfg (StackNetworkConfig): Network configuration.
        sweep_params (list): Parameters to sweep.
        num_experiments (int): Number of experiments per configuration.
        programs (dict): Progam that map a name to class.

    Returns:
        dict: A dictionary with the parameter mapping and computed results.
    """
    # Map parameter name to value.
    map_param = dict(zip(sweep_params, comb, strict=False))

    local_cfg = copy.deepcopy(cfg)
    update_cfg(local_cfg, map_param)

    res = run(config=local_cfg, programs=programs, num_times=num_experiments)
    last_program_results = res[len(programs) - 1]

    all_fid_results = [r[0] for r in last_program_results]
    all_time_results = [r[1] for r in last_program_results]

    avg_fid = np.mean(all_fid_results) * 100
    avg_time = np.mean(all_time_results)

    return {
        **map_param,
        "Fidelity Results": all_fid_results,
        "Simulation Time Results": all_time_results,
        "Average Fidelity (%)": avg_fid,
        "Average Simulation Time (ms)": avg_time,
    }
