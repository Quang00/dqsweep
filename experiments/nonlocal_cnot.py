"""
Nonlocal CNOT Quantum Experiment
--------------------------------

This module implements a nonlocal CNOT experiment between two parties:
Alice and Bob. The experiment utilizes quantum entanglement and classical
communication to perform a distributed CNOT gate.

The process consists of:
  - Alice generating EPR pairs, applying a CNOT gate, and measuring qubits.
  - Bob receiving the EPR pairs, applying corrections based on Alice’s
    measurements, and computing fidelity.
  - Classical communication between Alice and Bob to share measurement
    results and ensure proper corrections.

The results include fidelity computation and simulation time tracking
for performance evaluation.
"""

import numpy as np
from netqasm.sdk.classical_communication.socket import Socket
from netqasm.sdk.connection import BaseNetQASMConnection
from netqasm.sdk.epr_socket import EPRSocket
from netqasm.sdk.qubit import Qubit
from netsquid.qubits.dmutil import dm_fidelity
from netsquid.util.simtools import MILLISECOND, sim_time
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import get_qubit_state


# =============================================================================
# Alice's Program for Nonlocal CNOT
# =============================================================================
class AliceProgram(Program):
    """Implements Alice's side of the nonlocal CNOT experiment.

    Alice generates EPR pairs, applies a CNOT gate, performs measurements,
    and communicates results to Bob.

    Args:
        num_epr_rounds (int): Number of EPR rounds for the experiment.
    """

    PEER_NAME = "Bob"

    def __init__(self, num_epr_rounds: int):
        """Initializes Alice's program with the specified number of rounds.

        Args:
            num_epr_rounds (int): Number of EPR rounds in the experiment.
        """
        self._num_epr_rounds = num_epr_rounds

    @property
    def meta(self) -> ProgramMeta:
        """Defines metadata for Alice's CNOT program.

        Returns:
            ProgramMeta: Metadata including experiment name, sockets, and qubit limit.
        """
        return ProgramMeta(
            name="nonlocal_CNOT",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=1,
        )

    def run(self, context: ProgramContext):
        """Executes Alice's nonlocal CNOT experiment.

        In each round, Alice generates an EPR pair, applies a nonlocal CNOT,
        measures qubits, and communicates results to Bob.

        Args:
            context (ProgramContext): Provides network and connection details.

        """
        csocket = context.csockets[self.PEER_NAME]
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        for _ in range(self._num_epr_rounds):
            # Create an EPR pair
            a1_qubit = epr_socket.create_keep()[0]

            # Create a local qubit and apply an X gate
            alice_qubit = Qubit(connection)
            alice_qubit.X()

            # Perform a nonlocal CNOT operation
            alice_qubit.cnot(a1_qubit)

            # Measure Alice's entangled qubit
            a1_measurement = a1_qubit.measure()
            yield from connection.flush()

            # Send measurement result to Bob
            csocket.send(str(a1_measurement))

            # Receive Bob's measurement result
            b1_measurement = yield from csocket.recv()
            if b1_measurement == "1":
                alice_qubit.Z()

            # Free qubit and flush the connection
            alice_qubit.free()
            yield from connection.flush()

        return {}


# =============================================================================
# Bob's Program for Nonlocal CNOT
# =============================================================================
class BobProgram(Program):
    """Implements Bob's side of the nonlocal CNOT experiment.

    Bob receives EPR pairs, listens to Alice's measurement results,
    applies correction operations, and measures the final state to compute
    fidelity.

    Args:
        num_epr_rounds (int): Number of EPR rounds for the experiment.
    """

    PEER_NAME = "Alice"

    def __init__(self, num_epr_rounds: int):
        """Initializes Bob's program with the specified number of rounds.

        Args:
            num_epr_rounds (int): Number of EPR rounds in the experiment.
        """
        self._num_epr_rounds = num_epr_rounds

    @property
    def meta(self) -> ProgramMeta:
        """Defines metadata for Bob's CNOT program.

        Returns:
            ProgramMeta: Metadata including experiment name, sockets, and qubit limit.
        """
        return ProgramMeta(
            name="nonlocal_CNOT",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=1,
        )

    def run(self, context: ProgramContext):
        """Executes Bob's nonlocal CNOT experiment.

        In each round, Bob receives an EPR pair, listens for Alice's measurement
        results, applies corrections, and measures the final state.

        Args:
            context (ProgramContext): Provides network and connection details.

        Returns:
            tuple[list[float], list[float]]: A tuple containing lists of
            fidelities and simulation times.
        """
        csocket: Socket = context.csockets[self.PEER_NAME]
        epr_socket: EPRSocket = context.epr_sockets[self.PEER_NAME]
        connection: BaseNetQASMConnection = context.connection

        fidelities: list[float] = []
        simulation_times: list[float] = []

        for _ in range(self._num_epr_rounds):
            # Receive an EPR pair
            b1_qubit = epr_socket.recv_keep()[0]
            yield from connection.flush()

            # Create a local qubit as the target of the CNOT operation
            bob_qubit = Qubit(connection)

            # Receive Alice's measurement result
            a1_measurement = yield from csocket.recv()

            # Apply correction based on Alice's measurement
            if a1_measurement == "1":
                b1_qubit.X()

            # Perform CNOT operation and measure the control qubit
            b1_qubit.cnot(bob_qubit)
            b1_qubit.H()
            b1_measurement = b1_qubit.measure()
            yield from connection.flush()

            # Send measurement result back to Alice
            csocket.send(str(b1_measurement))

            # Compute fidelity of the final state
            dm_b = get_qubit_state(bob_qubit, "Bob", full_state=True)
            state_ref = np.array([0, 0, 0, 1], dtype=complex)
            dm_ref = np.outer(state_ref, np.conjugate(state_ref))
            fidelity = dm_fidelity(dm_b, dm_ref)
            fidelities.append(fidelity)

            # Free qubit and record simulation time
            bob_qubit.free()
            yield from connection.flush()
            simulation_times.append(sim_time(MILLISECOND))

        return fidelities, simulation_times
