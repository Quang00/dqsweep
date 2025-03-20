from typing import List

import numpy as np
from netqasm.sdk.qubit import Qubit

from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util.util import create_complete_graph_network
from utils.gates import ccz, toffoli


class TestGates:
    def _create_config(self, node_names: List[str]) -> StackNetworkConfig:
        """Create a network with the node's names given.

        Args:
            node_names (List[str]): List of the node names.

        Returns:
            StackNetworkConfig: Network configuration with the nodes.
        """
        return create_complete_graph_network(
            node_names=node_names,
            link_typ="perfect",
            link_cfg={},
            clink_typ="instant",
            clink_cfg=None,
            qdevice_typ="generic",
            qdevice_cfg=None,
        )

    class ToffoliTest(Program):
        def __init__(self, initial_state: tuple):
            """Initialize the Toffoli test program.

            Args:
                initial_state (tuple): Initial state in the form
                `(control1, control2, target)`.
            """
            self.initial_state = initial_state

        @property
        def meta(self) -> ProgramMeta:
            return ProgramMeta(
                name="toffoli_test", csockets=[], epr_sockets=[], max_qubits=3
            )

        def run(self, context: ProgramContext):
            connection = context.connection

            q0 = Qubit(connection)
            q1 = Qubit(connection)
            q2 = Qubit(connection)

            if self.initial_state[0]:
                q0.X()
            if self.initial_state[1]:
                q1.X()
            if self.initial_state[2]:
                q2.X()

            toffoli(q0, q1, q2)

            m0 = q0.measure()
            m1 = q1.measure()
            m2 = q2.measure()

            yield from connection.flush()

            return {"m0": int(m0), "m1": int(m1), "m2": int(m2)}

    class CCZTest(Program):
        def __init__(self, initial_state: tuple):
            """Initialize the CCZ test program.

            Args:
                initial_state (tuple): Initial state in the form
                `(control1, control2, target)`.
            """
            self.initial_state = initial_state

        @property
        def meta(self) -> ProgramMeta:
            return ProgramMeta(
                name="ccz_test", csockets=[], epr_sockets=[], max_qubits=3
            )

        def run(self, context: ProgramContext):
            connection = context.connection

            q0 = Qubit(connection)
            q1 = Qubit(connection)
            q2 = Qubit(connection)

            if self.initial_state[0]:
                q0.X()
            if self.initial_state[1]:
                q1.X()
            if self.initial_state[2]:
                q2.X()

            ccz(q0, q1, q2)

            m0 = q0.measure()
            m1 = q1.measure()
            m2 = q2.measure()

            yield from connection.flush()

            return {"m0": int(m0), "m1": int(m1), "m2": int(m2)}

    def test_toffoli_ket_1_ket_1_ket_0(self):
        """Test the Toffoli gate:

        - First control qubit is in state |1>.
        - Second control qubit is in state |1>.
        - Third qubit is the target qubit is in state |0>.

        At the end, target's qubit should be in state |1>.
        """
        config = self._create_config(["Alice"])
        program = self.ToffoliTest(initial_state=(1, 1, 0))
        all_results = run(config, {"Alice": program}, 10)
        result = all_results[0][0]

        np.testing.assert_equal(result["m0"], 1)
        np.testing.assert_equal(result["m1"], 1)
        np.testing.assert_equal(result["m2"], 1)

    def test_toffoli_ket_1_ket_0_ket_0(self):
        """Test the Toffoli gate:

         - First control qubit is in |1>.
         - Second control qubit is in |0>.
         - Target qubit is in |0>.

        At the end, target's qubit should be in state |1>.
        """
        config = self._create_config(["Alice"])
        program = self.ToffoliTest(initial_state=(1, 0, 0))
        all_results = run(config, {"Alice": program}, 10)
        result = all_results[0][0]

        np.testing.assert_equal(result["m0"], 1)
        np.testing.assert_equal(result["m1"], 0)
        np.testing.assert_equal(result["m2"], 0)

    def test_ccz_ket_1_ket_1_ket_0(self):
        """Test the CCZ gate:

        - First control qubit is in state |1>.
        - Second control qubit is in state |1>.
        - Third qubit is the target qubit is in state |0>.

        At the end, target's qubit should be in state |0>.
        """
        config = self._create_config(["Alice"])
        program = self.CCZTest(initial_state=(1, 1, 0))
        all_results = run(config, {"Alice": program}, 10)
        result = all_results[0][0]

        np.testing.assert_equal(result["m0"], 1)
        np.testing.assert_equal(result["m1"], 1)
        np.testing.assert_equal(result["m2"], 0)

    def test_ccz_ket_1_ket_1_ket_1(self):
        """Test the CCZ gate:

          - First control qubit is in |1>.
          - Second control qubit is in |1>.
          - Target qubit is in |1>.

        At the end, target's qubit should be in state |1>.
        """
        config = self._create_config(["Alice"])
        program = self.CCZTest(initial_state=(1, 1, 1))
        all_results = run(config, {"Alice": program}, 10)
        result = all_results[0][0]

        np.testing.assert_equal(result["m0"], 1)
        np.testing.assert_equal(result["m1"], 1)
        np.testing.assert_equal(result["m2"], 1)
