import os
import numpy as np

from experiments.run_simulation import sweep_parameters
from squidasm.run.stack.config import StackNetworkConfig


def test_basic_perfect_config(tmp_path):
    """Test sweep parameters with the perfect configuration.

    Args:
        tmp_path (str): Temporary path directory.
    """
    cfg = StackNetworkConfig.from_file("configurations/perfect.yaml")

    # Define sweep parameters and ranges.
    param_1 = "single_qubit_gate_depolar_prob"
    param_2 = "two_qubit_gate_depolar_prob"
    sweep_params = [param_1, param_2]
    ranges = ["0,1,2", "0,1,2"]
    experiment = "cnot"
    output_dir = str(tmp_path / "results")

    # Execute the parameter sweep.
    df = sweep_parameters(
        cfg=cfg,
        rounds=10,
        num_experiments=10,
        sweep_params=sweep_params,
        ranges=ranges,
        experiment=experiment,
        output_dir=output_dir,
    )

    # Check that the CSV file was created.
    output_file = os.path.join(output_dir, f"{experiment}_results.csv")
    np.testing.assert_equal(
        os.path.exists(output_file), True, err_msg="CSV file was not created."
    )

    # There should be 2 values per parameter, so 2*2 = 4 rows.
    np.testing.assert_equal(df.shape[0], 4, err_msg="Invalid number of rows.")

    # Verify the expected columns are present.
    expected_columns = [
        "single_qubit_gate_depolar_prob",
        "two_qubit_gate_depolar_prob",
        "Fidelity Results",
        "Simulation Time Results",
        "Average Fidelity",
        "Average Simulation Time (ms)",
    ]

    np.testing.assert_array_equal(df.columns, expected_columns)
