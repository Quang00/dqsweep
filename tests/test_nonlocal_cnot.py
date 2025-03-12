import numpy as np

from experiments.nonlocal_cnot import AliceProgram, BobProgram
from experiments.utils import run_simulation


def test_alice_ket_1_bob_ket_0():
    """Test the nonlocal CNOT gate:

    - Alice's control qubit is in state |1>.
    - Bob's target qubit is in state |0>.

    At the end, Bob's qubit should be in state |1>.
    """
    _, results = run_simulation(
        config="configurations/perfect.yaml",
        epr_rounds=10,
        num_times=10,
        alice_cls=AliceProgram,
        bob_cls=BobProgram,
    )

    # Compute the average fidelity.
    all_fid_results = [res[0] for res in results]
    avg_fidelity = np.mean(all_fid_results)

    # Check that the average fidelity is equal to 1.
    np.testing.assert_equal(avg_fidelity, 1.0)
