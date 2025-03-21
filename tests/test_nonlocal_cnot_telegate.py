import numpy as np

from experiments.nonlocal_cnot_telegate import AliceProgram, BobProgram
from utils.helper import simulate_and_compute_avg_fidelity
from squidasm.util.util import create_two_node_network


def test_alice_ket_1_bob_ket_0():
    """Test the nonlocal CNOT gate:

    - Alice's control qubit is in state |1>.
    - Bob's target qubit is in state |0>.

    At the end, Bob's qubit should be in state |1>.
    """
    config = create_two_node_network()
    epr_rounds = 10

    avg_fidelity = simulate_and_compute_avg_fidelity(
        config=config,
        programs={
            "Alice": AliceProgram(epr_rounds),
            "Bob": BobProgram(epr_rounds),
        },
    )

    # Check that the average fidelity is equal to 1.
    np.testing.assert_equal(avg_fidelity, 1.0)
