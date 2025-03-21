import numpy as np
import pytest

from experiments.pingpong import (
    AlicePingpongTeleportation,
    BobPingpongTeleportation,
)
from squidasm.util.util import create_two_node_network
from utils.helper import simulate_and_compute_avg_fidelity


@pytest.mark.parametrize("epr_rounds", [1, 3])
def test_various_hops(epr_rounds):
    """Test the pingpong experiment with different hop counts:

    For each case, Bob's qubit should be in the same state
    as the initial one sent by Alice.

    Args:
        epr_rounds (int): Total number of hops.
    """
    config = create_two_node_network()

    avg_fidelity = simulate_and_compute_avg_fidelity(
        config=config,
        programs={
            "Alice": AlicePingpongTeleportation(epr_rounds),
            "Bob": BobPingpongTeleportation(epr_rounds),
        },
    )

    # Average fidelity should be close to 1.
    np.testing.assert_almost_equal(
        avg_fidelity,
        1.0,
        err_msg=f"Should be 1.0 {epr_rounds} hops but got {avg_fidelity}",
    )
