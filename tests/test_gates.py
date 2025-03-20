from typing import Callable, List

import numpy as np
import pytest
from netqasm.sdk.qubit import Qubit

from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util.util import create_complete_graph_network
from utils.gates import ccz, n_qubit_controlled_u, toffoli


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
            """Initialize the Toffoli test class.

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
            """Initialize the CCZ test class.

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

    class NControlledUTest(Program):
        def __init__(
            self,
            ctrls_init_state: tuple,
            tgt_init_state: int,
            controlled_u_gate: Callable[[Qubit, Qubit], None],
        ):
            """Initialize the n-controlled U test class.

            Args:
                ctrls_init_state (tuple): Initial state for the control qubits.
                tgt_init_state (int): Initial state of the target qubit.
                controlled_u_gate (Callable): The controlled-U gate (e.g.
                for CNOT: `lambda control, target: control.cnot(target)`,
                for CZ: `lambda control, target: control.cz(target)`)
            """
            self.ctrls_init_state = ctrls_init_state
            self.tgt_init_state = tgt_init_state
            self.controlled_u_gate = controlled_u_gate

        @property
        def meta(self) -> ProgramMeta:
            return ProgramMeta(
                name="ncontrolled_u_test",
                csockets=[],
                epr_sockets=[],
                max_qubits=len(self.ctrls_init_state) + 1,
            )

        def run(self, context: ProgramContext):
            connection = context.connection
            n = len(self.ctrls_init_state)
            controls_qubit = [Qubit(connection) for _ in range(n)]
            target = Qubit(connection)

            for i, state in enumerate(self.ctrls_init_state):
                if state == 1:
                    controls_qubit[i].X()

            if self.tgt_init_state == 1:
                target.X()

            n_qubit_controlled_u(
                controls_qubit, context, self.controlled_u_gate, target
            )

            m_target = target.measure()

            yield from connection.flush()

            return {"target": int(m_target)}

    @pytest.mark.parametrize(
        "initial_state, expected",
        [
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, 1), (0, 0, 1)),
            ((0, 1, 0), (0, 1, 0)),
            ((0, 1, 1), (0, 1, 1)),
            ((1, 0, 0), (1, 0, 0)),
            ((1, 0, 1), (1, 0, 1)),
            ((1, 1, 0), (1, 1, 1)),
            ((1, 1, 1), (1, 1, 0)),
        ],
    )
    def test_toffoli_gate(self, initial_state, expected):
        """Parametrized test for the Toffoli gate in the form:
        `(|control1>, |control2>, |target>) -> (m0, m1, m2)`.
        """
        config = self._create_config(["Alice"])
        program = self.ToffoliTest(initial_state=initial_state)
        all_results = run(config, {"Alice": program}, num_times=10)
        result = all_results[0][0]

        np.testing.assert_equal(result["m0"], expected[0])
        np.testing.assert_equal(result["m1"], expected[1])
        np.testing.assert_equal(result["m2"], expected[2])

    @pytest.mark.parametrize(
        "initial_state, expected",
        [
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, 1), (0, 0, 1)),
            ((0, 1, 0), (0, 1, 0)),
            ((0, 1, 1), (0, 1, 1)),
            ((1, 0, 0), (1, 0, 0)),
            ((1, 0, 1), (1, 0, 1)),
            ((1, 1, 0), (1, 1, 0)),
            ((1, 1, 1), (1, 1, 1)),
        ],
    )
    def test_ccz_gate(self, initial_state, expected):
        """Parametrized test for the CCZ gate in the form:
        `(|control1>, |control2>, |target>) -> (m0, m1, m2)`.
        """
        config = self._create_config(["Alice"])
        program = self.CCZTest(initial_state=initial_state)
        all_results = run(config, {"Alice": program}, num_times=10)
        result = all_results[0][0]

        np.testing.assert_equal(result["m0"], expected[0])
        np.testing.assert_equal(result["m1"], expected[1])
        np.testing.assert_equal(result["m2"], expected[2])

    @pytest.mark.parametrize(
        "ctrls_init_state, tgt_init_state, controlled_u_gate, expected",
        [
            # CNOT Gate
            ((0, 0, 0), 0, lambda ctrl, tgt: ctrl.cnot(tgt), 0),
            ((0, 0, 1), 1, lambda ctrl, tgt: ctrl.cnot(tgt), 1),
            ((0, 1, 0), 0, lambda ctrl, tgt: ctrl.cnot(tgt), 0),
            ((0, 1, 1), 1, lambda ctrl, tgt: ctrl.cnot(tgt), 1),
            ((1, 1, 0), 0, lambda ctrl, tgt: ctrl.cnot(tgt), 0),
            ((1, 1, 1), 0, lambda ctrl, tgt: ctrl.cnot(tgt), 1),
            ((1, 1, 1), 1, lambda ctrl, tgt: ctrl.cnot(tgt), 0),
            (((0,) * 5), 0, lambda ctrl, tgt: ctrl.cnot(tgt), 0),
            (((0,) * 5), 1, lambda ctrl, tgt: ctrl.cnot(tgt), 1),
            (((1,) * 5), 0, lambda ctrl, tgt: ctrl.cnot(tgt), 1),
            (((1,) * 5), 1, lambda ctrl, tgt: ctrl.cnot(tgt), 0),
            (((1,) * 4 + (0,)), 1, lambda ctrl, tgt: ctrl.cnot(tgt), 1),
            (((1,) + (0,) * 4), 0, lambda ctrl, tgt: ctrl.cnot(tgt), 0),
            # CZ Gate
            ((0, 0, 0), 0, lambda ctrl, tgt: ctrl.cphase(tgt), 0),
            ((0, 0, 1), 1, lambda ctrl, tgt: ctrl.cphase(tgt), 1),
            ((0, 1, 0), 0, lambda ctrl, tgt: ctrl.cphase(tgt), 0),
            ((0, 1, 1), 1, lambda ctrl, tgt: ctrl.cphase(tgt), 1),
            ((1, 1, 0), 0, lambda ctrl, tgt: ctrl.cphase(tgt), 0),
            ((1, 1, 1), 0, lambda ctrl, tgt: ctrl.cphase(tgt), 0),
            ((1, 1, 1), 1, lambda ctrl, tgt: ctrl.cphase(tgt), 1),
            (((0,) * 5), 0, lambda ctrl, tgt: ctrl.cphase(tgt), 0),
            (((0,) * 5), 1, lambda ctrl, tgt: ctrl.cphase(tgt), 1),
            (((1,) * 5), 0, lambda ctrl, tgt: ctrl.cphase(tgt), 0),
            (((1,) * 5), 1, lambda ctrl, tgt: ctrl.cphase(tgt), 1),
            (((1,) * 4 + (0,)), 1, lambda ctrl, tgt: ctrl.cphase(tgt), 1),
            (((1,) + (0,) * 4), 0, lambda ctrl, tgt: ctrl.cphase(tgt), 0),
        ],
    )
    def test_n_controlled_u_states(
        self, ctrls_init_state, tgt_init_state, controlled_u_gate, expected
    ):
        """Parametrized test for the n-controlled U gate in the form:
        `(|control0>, ... |controln>), |target>, controlled-U -> target_meas`.
        """
        config = self._create_config(["Alice"])
        program = self.NControlledUTest(
            ctrls_init_state=ctrls_init_state,
            tgt_init_state=tgt_init_state,
            controlled_u_gate=controlled_u_gate,
        )
        all_results = run(config, {"Alice": program}, num_times=10)
        result = all_results[0][0]

        # Check the final measurement of the target qubit
        np.testing.assert_equal(result["target"], expected)
