"""
Distributed Grover Quantum Experiment
--------------------------------
This file implements the distributed Grover on 2 qubits,
between two nodes (Alice and Bob).
"""

import numpy as np
from netqasm.sdk.qubit import Qubit
from netsquid.qubits.dmutil import dm_fidelity
from netsquid.util.simtools import MILLISECOND, sim_time

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import get_qubit_state
from squidasm.util.routines import (
    distributed_CPhase_control,
    distributed_CPhase_target
)


# =============================================================================
# Alice's Program for distributed Grover on 2 qubits
# =============================================================================
class AliceDGrover2(Program):
    """
    Implements Alice's side of distributed Grover on 2 qubits.

    Args:
        num_epr_rounds (int):  Number of EPR rounds for the experiment.
    """

    PEER_NAME = "Bob"

    def __init__(self, num_epr_rounds: int):
        """Initializes Alice's program with the specified number of rounds."""
        self._num_epr_rounds = num_epr_rounds

    @property
    def meta(self) -> ProgramMeta:
        """Defines metadata for Alice's distributed Grover program.

        Returns:
            ProgramMeta: Metadata -> experiment name, sockets, qubit limit.
        """
        return ProgramMeta(
            name="dqft",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        """
        Executes Alice's part of distributed Grover on 2 qubits.

        """

        for _ in range(self._num_epr_rounds):

            a_q = Qubit(context.connection)
            a_q.H()

            yield from distributed_CPhase_control(context, self.PEER_NAME, a_q)
            yield from context.connection.flush()

            a_q.H()
            a_q.X()
            yield from distributed_CPhase_control(context, self.PEER_NAME, a_q)
            yield from context.connection.flush()
            a_q.X()
            a_q.H()

            a_q.free()
            yield from context.connection.flush()

        return {}


# =============================================================================
# Bob's Program for distributed Grover on 2 qubits
# =============================================================================
class BobDGrover2(Program):
    """
    Implements Bob's side of distributed Grover on 2 qubits.

    Args:
        num_epr_rounds (int): Number of EPR rounds for the experiment.
    """

    PEER_NAME = "Alice"

    def __init__(self, num_epr_rounds: int):
        """Initializes Bob's program with the specified number of rounds."""
        self._num_epr_rounds = num_epr_rounds
        self.fidelities: list[float] = []
        self.simulation_times: list[float] = []

    @property
    def meta(self) -> ProgramMeta:
        """Defines metadata for Bob's distributed Grover program.

        Returns:
            ProgramMeta: Metadata -> experiment name, sockets, qubit limit.
        """
        return ProgramMeta(
            name="dqft",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        """
        Executes Bob's part of distributed Grover on 2 qubits.

        Returns:
            tuple[list[float], list[float]]: The fidelity and simulation time
            lists for each round.
        """

        for _ in range(self._num_epr_rounds):
            b_q = Qubit(context.connection)
            b_q.H()

            yield from distributed_CPhase_target(context, self.PEER_NAME, b_q)
            yield from context.connection.flush()

            b_q.H()
            b_q.X()
            yield from distributed_CPhase_target(context, self.PEER_NAME, b_q)
            yield from context.connection.flush()
            b_q.X()
            b_q.H()

            yield from context.connection.flush()
            dm_b = get_qubit_state(b_q, "Bob", full_state=True)
            print(dm_b)
            state_ref = np.array([0, 0, 0, 1], dtype=complex)
            dm_ref = np.outer(state_ref, np.conjugate(state_ref))
            fidelity = dm_fidelity(dm_b, dm_ref, dm_check=False)
            self.fidelities.append(fidelity)
            self.simulation_times.append(sim_time(MILLISECOND))

            b_q.free()
            yield from context.connection.flush()

        return self.fidelities, self.simulation_times
