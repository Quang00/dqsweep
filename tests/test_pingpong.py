import numpy as np
import pytest

from experiments.pingpong import (
    AlicePingpongTeleportation,
    BobPingpongTeleportation,
)
from experiments.utils import run_simulation


@pytest.mark.parametrize("epr_rounds", [1, 3])
def test_various_hops(epr_rounds):
    """Test the pingpong experiment with different hop counts:

    For each case, Bob's qubit should be in the same state
    as the initial one sent by Alice.

    Args:
        epr_rounds (int): Total number of hops.
    """
    _, results = run_simulation(
        config="configurations/perfect.yaml",
        epr_rounds=epr_rounds,
        num_times=10,
        classes={
            "Alice": AlicePingpongTeleportation,
            "Bob": BobPingpongTeleportation,
        }
    )

    # Compute the average fidelity across experiments.
    all_fid_results = [res[0] for res in results]
    avg_fidelity = np.mean(all_fid_results)

    # Average fidelity should be close to 1.
    np.testing.assert_almost_equal(
        avg_fidelity,
        1.0,
        err_msg=f"Should be 1.0 {epr_rounds} hops but got {avg_fidelity}",
    )


@pytest.mark.parametrize("epr_rounds", [0, 2])
def test_even_hops(epr_rounds):
    """Test the pingpong experiment with even hop counts:

    For each case, it should raise an error because to complete
    a pingpong exchange, this number should be odd.

    Args:
        epr_rounds (int): Total number of hops.
    """
    with pytest.raises(ValueError):
        run_simulation(
            config="configurations/perfect.yaml",
            epr_rounds=epr_rounds,
            num_times=10,
            classes={
                "Alice": AlicePingpongTeleportation,
                "Bob": BobPingpongTeleportation,
            }
        )
