import numpy as np

from experiments.dgrover_2 import AliceDGrover2, BobDGrover2
from experiments.utils import run_simulation
from squidasm.util.util import create_two_node_network


def test_alice_ket_0_bob_ket_0():
    """Test the distributed quantum grover with:

    - Alice's qubit is in state |0>.
    - Bob's qubit is in state |0>.

    At the end, the qubit state should be in state: -|11>.
    """
    config = create_two_node_network()

    avg_fidelity = run_simulation(
        config=config,
        programs={
            "Alice": AliceDGrover2,
            "Bob": BobDGrover2,
        },
    )

    # Check that the average fidelity is equal to 1.
    np.testing.assert_equal(avg_fidelity, 1.0)
