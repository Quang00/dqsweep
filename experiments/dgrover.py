"""
Distributed Grover
---------------------------------

This module implements the distributed Grover on n qubits,
between n parties.
"""

from netqasm.sdk.qubit import Qubit
from netsquid.util.simtools import MILLISECOND, sim_time

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from utils.gates import multi_controlled_u_gate
from utils.routines import (
    distributed_n_qubit_controlled_u_control,
    distributed_n_qubit_controlled_u_target,
)


# =============================================================================
# Grover's Control for distributed Grover on n qubits
# =============================================================================
class GroverControl(Program):
    """Implements Control's side of distributed Grover on n qubits."""

    def __init__(self, target_peer: str, num_epr_rounds: int):
        self.target_peer = target_peer
        self._num_epr_rounds = num_epr_rounds

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="dgrover",
            csockets=[self.target_peer],
            epr_sockets=[self.target_peer],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        for _ in range(self._num_epr_rounds):
            # --- Initialization ---
            ctrl_qubit = Qubit(context.connection)
            ctrl_qubit.H()

            # --- Oracle ---
            yield from distributed_n_qubit_controlled_u_control(
                context, self.target_peer, ctrl_qubit
            )

            # --- Diffusion ---
            ctrl_qubit.H()
            ctrl_qubit.X()
            yield from distributed_n_qubit_controlled_u_control(
                context, self.target_peer, ctrl_qubit
            )
            ctrl_qubit.X()
            ctrl_qubit.H()

            meas = ctrl_qubit.measure()
            yield from context.connection.flush()

            # --- Round completed --
            context.csockets[self.target_peer].send(int(meas))
            yield from context.connection.flush()


# =============================================================================
# Grover's Target for distributed Grover on n qubits
# =============================================================================
class GroverTarget(Program):
    """Implements Target's side of distributed Grover on n qubits."""

    def __init__(self, control_peers: list, num_epr_rounds: int):
        self.control_peers = control_peers
        self._num_epr_rounds = num_epr_rounds
        self.fidelities = []
        self.simulation_times = []

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="dgrover",
            csockets=self.control_peers,
            epr_sockets=self.control_peers,
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        for _ in range(self._num_epr_rounds):
            # --- Initialization ---
            target_qubit = Qubit(context.connection)
            target_qubit.H()

            # --- Oracle ---
            yield from distributed_n_qubit_controlled_u_target(
                context,
                self.control_peers,
                target_qubit,
                multi_controlled_u_gate(
                    context, lambda control, target: control.cphase(target)
                ),
            )

            # --- Diffusion ---
            target_qubit.H()
            target_qubit.X()
            yield from distributed_n_qubit_controlled_u_target(
                context,
                self.control_peers,
                target_qubit,
                multi_controlled_u_gate(
                    context, lambda control, target: control.cphase(target)
                ),
            )
            target_qubit.X()
            target_qubit.H()
            yield from context.connection.flush()

            # --- Round completed ---
            all_msg = []
            for control_peer in self.control_peers:
                msg = yield from context.csockets[control_peer].recv()
                all_msg.append(msg)

            if len(all_msg) == len(self.control_peers):
                target_meas = target_qubit.measure()
                yield from context.connection.flush()
                all_msg.append(target_meas)
                fidelity = 0 if 0 in all_msg else 1

                self.fidelities.append(fidelity)
                self.simulation_times.append(sim_time(MILLISECOND))

            yield from context.connection.flush()

        return self.fidelities, self.simulation_times
