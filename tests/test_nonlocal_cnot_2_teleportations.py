import numpy as np

from experiments.nonlocal_cnot_2_teleportations import (
    Alice2Teleportations,
    Bob2Teleportations,
)
from experiments.utils import run_simulation
from squidasm.util.util import create_two_node_network


def test_alice_ket_1_bob_ket_0():
    """Test the nonlocal CNOT gate using the approach with 2 teleportations:

    - Alice's control qubit is in state |1>.
    - Bob's target qubit is in state |0>.

    At the end, Bob's qubit should be in state |1>.
    """
    config = create_two_node_network()

    avg_fidelity = run_simulation(
        config=config,
        programs={
            "Alice": Alice2Teleportations,
            "Bob": Bob2Teleportations,
        },
    )

    # Check that the average fidelity is equal to 1.
    np.testing.assert_equal(avg_fidelity, 1.0)
