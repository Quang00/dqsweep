import pytest
import numpy as np

from experiments.dgrover import GroverControl, GroverTarget
from squidasm.util.util import create_complete_graph_network
from utils.helper import simulate_and_compute_avg_fidelity


@pytest.mark.parametrize(
    "node_names, fidelity_threshold",
    [
        (["Alice", "Bob", "Charlie"], 0.6),
        (["Alice", "Bob", "Charlie", "David"], 0.4),
    ],
)
def test_distributed_quantum_grover(node_names, fidelity_threshold):
    """Test the distributed Grover for different number of nodes.
    For each cases, the configuration is perfect. The first N-1 nodes are
    the controls nodes and the last node is the target. The average fidelity
    after the simulations should be above a certain threshold.
    """
    config = create_complete_graph_network(
        node_names=node_names,
        link_typ="perfect",
        link_cfg={},
        clink_typ="instant",
        clink_cfg=None,
        qdevice_typ="generic",
        qdevice_cfg=None,
    )
    rounds = 10
    ctrl_peers = node_names[:-1]
    trgt_peer = node_names[-1]

    programs = {name: GroverControl(trgt_peer, rounds) for name in ctrl_peers}
    programs[trgt_peer] = GroverTarget(ctrl_peers, rounds)

    avg_fidelity = simulate_and_compute_avg_fidelity(config, programs)

    np.testing.assert_(avg_fidelity >= fidelity_threshold)
