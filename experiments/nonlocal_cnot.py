"""
Nonlocal CNOT Gate
------------------

This module implements a nonlocal CNOT gate between two parties:
Alice and Bob using one ebit and one bit in each direction. The
implementation is from the paper "Optimal local implementation
of nonlocal quantum gates" (2000), Eisert, Jens et al.

The initial control qubit state of Alice is |1>, the initial target
qubit state of Bob is |0>. After the distributed CNOT gate, the qubit
of Bob should be |1>. This is verified by computing the fidelity
between the density output matrix and the expected one.
"""

import numpy as np

from netqasm.sdk.classical_communication.socket import Socket
from netqasm.sdk.connection import BaseNetQASMConnection
from netqasm.sdk.epr_socket import EPRSocket
from netqasm.sdk.qubit import Qubit

from netsquid.util.simtools import MILLISECOND, sim_time

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta

from experiments.utils import compute_fidelity


# =============================================================================
# Alice's Program for Nonlocal CNOT Gate
# =============================================================================
class AliceProgram(Program):
    """Implements Alice's side of the nonlocal CNOT gate.
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
        """Defines metadata for Alice's program.

        Returns:
            ProgramMeta: Experiment name, sockets, qubit limit.
        """
        return ProgramMeta(
            name="nonlocal_CNOT",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        """Executes Alice's part.

        Args:
            context (ProgramContext): Network and connection details.

        """
        csocket = context.csockets[self.PEER_NAME]
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        for _ in range(self._num_epr_rounds):
            # Create an EPR pair
            a1_qubit = epr_socket.create_keep()[0]

            # Create a local qubit which is the control qubit of the CNOT gate
            alice_qubit = Qubit(connection)
            alice_qubit.X()
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

            # Measure qubit and flush the connection
            alice_qubit.measure()
            yield from connection.flush()


# =============================================================================
# Bob's Program for Nonlocal CNOT Gate
# =============================================================================
class BobProgram(Program):
    """Implements Bob's side of the nonlocal CNOT gate.
    """

    PEER_NAME = "Alice"

    def __init__(self, num_epr_rounds: int):
        """Initializes Bob's program with the given number of rounds.

        Args:
            num_epr_rounds (int): Number of EPR rounds.
        """
        self._num_epr_rounds = num_epr_rounds

    @property
    def meta(self) -> ProgramMeta:
        """Defines metadata for Bob's program.

        Returns:
            ProgramMeta: Experiment name, sockets, qubit limit.
        """
        return ProgramMeta(
            name="nonlocal_CNOT",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        """Executes Bob's part.

        Args:
            context (ProgramContext): Network and connection details.

        Returns:
            list[tuple[list[float], list[float]]]: A list of tuple containing
            lists of fidelities and simulation times.
        """
        csocket: Socket = context.csockets[self.PEER_NAME]
        epr_socket: EPRSocket = context.epr_sockets[self.PEER_NAME]
        connection: BaseNetQASMConnection = context.connection

        fidelities: list[float] = []
        simulation_times: list[float] = []

        for _ in range(self._num_epr_rounds):
            # Receive the EPR pair
            b1_qubit = epr_socket.recv_keep()[0]
            yield from connection.flush()

            # Create a local qubit which is the target qubit of the CNOT gate
            bob_qubit = Qubit(connection)

            # Receive Alice's measurement result
            a1_measurement = yield from csocket.recv()

            # Apply X gate based on Alice's measurement
            if a1_measurement == "1":
                b1_qubit.X()

            # Perform CNOT operation and measure the control qubit
            b1_qubit.cnot(bob_qubit)
            b1_qubit.H()
            b1_measurement = b1_qubit.measure()
            yield from connection.flush()

            # Send measurement result to Alice
            csocket.send(str(b1_measurement))

            # Compare density matrices with the expected state
            state_ref = np.array([0, 0, 0, 1], dtype=complex)
            fidelity = compute_fidelity(bob_qubit, "Bob", state_ref)

            bob_qubit.measure()

            # Store the fidelity and simulation time results
            fidelities.append(fidelity)
            simulation_times.append(sim_time(MILLISECOND))

            yield from connection.flush()

        return fidelities, simulation_times
