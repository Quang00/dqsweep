import numpy as np

from experiments.nonlocal_cnot_2_teleportations import (
    Alice2Teleportations,
    Bob2Teleportations,
)
from experiments.utils import run_simulation


def test_alice_ket_1_bob_ket_0():
    """
    Test the nonlocal CNOT gate using the approach with 2 teleportations:

    - Alice's control qubit is in state |1>.
    - Bob's target qubit is in state |0>.

    At the end, Bob's qubit should be in state |1>.
    """
    _, results = run_simulation(
        config="configurations/perfect.yaml",
        epr_rounds=10,
        num_times=10,
        alice_cls=Alice2Teleportations,
        bob_cls=Bob2Teleportations,
    )

    # Compute the average fidelity.
    all_fid_results = [res[0] for res in results]
    avg_fidelity = np.mean(all_fid_results)

    # Check that the average fidelity is equal to 1.
    assert avg_fidelity == 1.0
