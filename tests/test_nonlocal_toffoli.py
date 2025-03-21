import numpy as np

from experiments.nonlocal_toffoli import (
    AliceToffoli,
    BobToffoli,
    CharlieToffoli
)
from squidasm.util.util import create_complete_graph_network
from utils.helper import simulate_and_compute_avg_fidelity


def test_alice_ket_1_bob_ket_1_charlie_ket_0():
    """Test the nonlocal Toffoli gate:

    - Alice's control qubit is in state |1>.
    - Bob's target qubit is in state |1>.
    - Charlie's target qubit is in state |0>.

    At the end, Charlie's qubit should be in state |1>.
    """
    config = create_complete_graph_network(
        node_names=["Alice", "Bob", "Charlie"],
        link_typ="perfect",
        link_cfg={},
        clink_typ="instant",
        clink_cfg=None,
        qdevice_typ="generic",
        qdevice_cfg=None,
    )
    epr_rounds = 10

    avg_fidelity = simulate_and_compute_avg_fidelity(
        config=config,
        programs={
            "Alice": AliceToffoli(epr_rounds),
            "Bob": BobToffoli(epr_rounds),
            "Charlie": CharlieToffoli(epr_rounds),
        },
    )

    # Check that the average fidelity is equal to 1.
    np.testing.assert_equal(avg_fidelity, 1.0)
