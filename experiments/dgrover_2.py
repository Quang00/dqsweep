"""
Distributed Grover On Two Qubits
---------------------------------

This module implements the distributed Grover on 2 qubits,
between two parties: Alice and Bob.

Alice's qubit is in state |0> and Bob's qubit should be in state |+>
if the inital ping-pong teleportation was performed without noise. After
the distributed Grover, the expected state should be -|11>, which is assess
by the expected density matrix compared to the output density matrix that gives
the fidelity.

Note: The implementation incorporates an initial ping-pong teleportation with
`n = 3` where Alice initiates and sends the state of Bob's qubit. Then, Bob
receives and sends it back to Alice. Finally, she sends once again the qubit
and then the distributed Grover starts with this back-and-forth qubit.
"""

import numpy as np
from netqasm.sdk.qubit import Qubit
from netsquid.util.simtools import MILLISECOND, sim_time

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util.routines import (
    distributed_CPhase_control,
    distributed_CPhase_target
)

from utils.helper import compute_fidelity
from utils.routines import pingpong_initiator, pingpong_responder


# =============================================================================
# Alice's Program for distributed Grover on 2 qubits
# =============================================================================
class AliceDGrover2(Program):
    """Implements Alice's side of distributed Grover on 2 qubits.
    """

    PEER_NAME = "Bob"

    def __init__(self, num_epr_rounds: int):
        """Initializes Alice's program with the given number of rounds.

        Args:
            num_epr_rounds (int): Number of EPR rounds.
        """
        self._num_epr_rounds = num_epr_rounds

    @property
    def meta(self) -> ProgramMeta:
        """Defines metadata for Alice's distributed Grover program.

        Returns:
            ProgramMeta: Experiment name, sockets, qubit limit.
        """
        return ProgramMeta(
            name="dgrover2",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        """Executes Alice's part of distributed Grover on 2 qubits.

        Args:
            context (ProgramContext): Network and connection details.
        """

        for _ in range(self._num_epr_rounds):
            # --- Initialization ---
            a_q = Qubit(context.connection)
            a_q.H()
            b_q = Qubit(context.connection)
            b_q.H()
            yield from pingpong_initiator(b_q, context, self.PEER_NAME)

            # --- Oracle ---
            yield from distributed_CPhase_control(context, self.PEER_NAME, a_q)

            # --- Diffusion ---
            a_q.H()
            a_q.X()
            yield from distributed_CPhase_control(context, self.PEER_NAME, a_q)
            a_q.X()
            a_q.H()
            yield from context.connection.flush()

            # --- Round completed --
            context.csockets[self.PEER_NAME].send("ACK")
            a_q.measure()
            yield from context.connection.flush()


# =============================================================================
# Bob's Program for distributed Grover on 2 qubits
# =============================================================================
class BobDGrover2(Program):
    """Implements Bob's side of distributed Grover on 2 qubits.
    """

    PEER_NAME = "Alice"

    def __init__(self, num_epr_rounds: int):
        """Initializes Bob's program with the given number of rounds.

        Args:
            num_epr_rounds (int):  Number of EPR rounds.
        """
        self._num_epr_rounds = num_epr_rounds
        self.fidelities = []
        self.simulation_times = []

    @property
    def meta(self) -> ProgramMeta:
        """Defines metadata for Bob's distributed Grover program.

        Returns:
            ProgramMeta: Experiment name, sockets, qubit limit.
        """
        return ProgramMeta(
            name="dgrover2",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        """Executes Bob's part of distributed Grover on 2 qubits.

        Args:
            context (ProgramContext): Network and connection details.

        Returns:
            list[tuple[list[float], list[float]]]: A list of tuple containing
            lists of fidelities and simulation times.
        """

        for _ in range(self._num_epr_rounds):
            # --- Initialization ---
            b_q = yield from pingpong_responder(context, self.PEER_NAME)

            # --- Oracle ---
            yield from distributed_CPhase_target(context, self.PEER_NAME, b_q)

            # --- Diffusion ---
            b_q.H()
            b_q.X()
            yield from distributed_CPhase_target(context, self.PEER_NAME, b_q)
            b_q.X()
            b_q.H()
            yield from context.connection.flush()

            # --- Round completed ---
            msg = yield from context.csockets[self.PEER_NAME].recv()
            if msg == "ACK":
                state_ref = np.array([0, 0, 0, 1], dtype=complex)
                fidelity = compute_fidelity(b_q, "Bob", state_ref)

                b_q.measure()

                self.fidelities.append(fidelity)
                self.simulation_times.append(sim_time(MILLISECOND))

            yield from context.connection.flush()

        return self.fidelities, self.simulation_times
