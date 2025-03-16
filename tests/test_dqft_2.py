import numpy as np

from experiments.dqft_2 import AliceDQFT2, BobDQFT2
from experiments.utils import run_simulation


def test_alice_ket_0_bob_ket_0():
    """Test the distributed quantum fourier transform with:

    - Alice's qubit is in state |0>.
    - Bob's qubit is in state |0>.

    At the end, the qubit state should be in the following
    fourier basis:  1/2 (|00> + |01> + |10> + |11>).
    """
    _, results = run_simulation(
        config="configurations/perfect.yaml",
        epr_rounds=10,
        num_times=10,
        classes={
            "Alice": AliceDQFT2,
            "Bob": BobDQFT2,
        }
    )

    # Compute the average fidelity.
    all_fid_results = [res[0] for res in results]
    avg_fidelity = np.mean(all_fid_results)

    # Check that the average fidelity is equal to 1.
    np.testing.assert_equal(avg_fidelity, 1.0)
