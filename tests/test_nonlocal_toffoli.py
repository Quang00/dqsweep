import numpy as np

from experiments.nonlocal_toffoli import (
    AliceToffoli,
    BobToffoli,
    CharlieToffoli
)
from experiments.utils import run_simulation


def test_alice_ket_1_bob_ket_1_charlie_ket_0():
    """Test the nonlocal Toffoli gate:

    - Alice's control qubit is in state |1>.
    - Bob's target qubit is in state |1>.
    - Charlie's target qubit is in state |0>.

    At the end, Charlie's qubit should be in state |1>.
    """
    _, _, results = run_simulation(
        config="configurations/3_nodes.yaml",
        epr_rounds=10,
        num_times=10,
        classes={
            "Alice": AliceToffoli,
            "Bob": BobToffoli,
            "Charlie": CharlieToffoli,
        }
    )

    # Compute the average fidelity.
    all_fid_results = [res[0] for res in results]
    avg_fidelity = np.mean(all_fid_results)

    # Check that the average fidelity is equal to 1.
    np.testing.assert_equal(avg_fidelity, 1.0)
