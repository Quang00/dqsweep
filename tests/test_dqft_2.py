import numpy as np

from experiments.dqft_2 import AliceDQFT2, BobDQFT2
from squidasm.util.util import create_two_node_network
from utils.helper import simulate_and_compute_avg_fidelity


def test_alice_ket_0_bob_ket_0():
    """Test the distributed quantum fourier transform with:

    - Alice's qubit is in state |0>.
    - Bob's qubit is in state |0>.

    At the end, the qubit state should be in the following
    fourier basis:  1/2 (|00> + |01> + |10> + |11>).
    """
    config = create_two_node_network()
    epr_rounds = 10

    avg_fidelity = simulate_and_compute_avg_fidelity(
        config=config,
        programs={
            "Alice": AliceDQFT2(epr_rounds),
            "Bob": BobDQFT2(epr_rounds),
        },
    )

    # Check that the average fidelity is equal to 1.
    np.testing.assert_equal(avg_fidelity, 1.0)
