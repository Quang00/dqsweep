"""
Nonlocal CNOT Gate With Two Teleportations
------------------------------------------

This module implements a quantum teleportation experiment using a nonlocal
CNOT gate between two parties: Alice and Bob. Alice prepares and sends
entangled qubits while Bob receives them, applies corrections, and measures
the final state to assess fidelity.
"""

import numpy as np

from netqasm.sdk.classical_communication.socket import Socket
from netqasm.sdk.connection import BaseNetQASMConnection
from netqasm.sdk.epr_socket import EPRSocket
from netqasm.sdk.qubit import Qubit

from netsquid.util.simtools import MILLISECOND, sim_time

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util.routines import teleport_recv, teleport_send

from experiments.utils import compute_fidelity


# =============================================================================
# Alice's Teleportation Program
# =============================================================================
class AliceTeleportation(Program):
    """Implements Alice's side of a nonlocal CNOT teleportation experiment.

    Alice generates EPR pairs, applies a CNOT gate, performs measurements,
    and sends results to Bob using classical communication.

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
        """Defines metadata for Alice's teleportation program.

        Returns:
            ProgramMeta: Metadata -> experiment name, sockets, qubit limit.
        """
        return ProgramMeta(
            name="teleportation",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=1,
        )

    def run(self, context: ProgramContext):
        """Executes Alice's teleportation process.

        In each round, Alice generates an EPR pair, applies a nonlocal CNOT
        operation, performs measurements, and sends results to Bob via a
        classical channel.

        Args:
            context (ProgramContext): Provides network and connection details.

        """
        csocket = context.csockets[self.PEER_NAME]
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        for _ in range(self._num_epr_rounds):
            # Create an EPR pair
            a1_qubit = epr_socket.create_keep()[0]

            # Create a local control qubit and apply an X gate
            alice_qubit = Qubit(connection)
            alice_qubit.X()

            # Perform a nonlocal CNOT operation
            alice_qubit.cnot(a1_qubit)
            alice_qubit.H()

            # Measure both qubits
            a1_measurement = a1_qubit.measure()
            alice_measurement = alice_qubit.measure()
            yield from connection.flush()

            # Send measurement results to Bob
            csocket.send(str(a1_measurement))
            csocket.send(str(alice_measurement))

            # Alice receive the state of Bob after the CNOT gate was applied
            teleport_recv(context, self.PEER_NAME)
            yield from connection.flush()

        return {}


# =============================================================================
# Bob's Teleportation Program
# =============================================================================
class BobTeleportation(Program):
    """Implements Bob's side of the nonlocal CNOT teleportation experiment.

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
        self.fidelities: list[float] = []
        self.simulation_times: list[float] = []

    @property
    def meta(self) -> ProgramMeta:
        """Defines metadata for Bob's teleportation program.

        Returns:
            ProgramMeta: Metadata -> experiment name, sockets, qubit limit.
        """
        return ProgramMeta(
            name="teleportation",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=1,
        )

    def run(self, context: ProgramContext):
        """Executes Bob's teleportation process.

        In each round, Bob receives an EPR pair, wait for Alice's measurement
        results, applies corrections, and measures the final state.

        Args:
            context (ProgramContext): Provides network and connection details.

        Returns:
            tuple[list[float], list[float]]: A generator
            yielding control back to the scheduler and eventually returning a
            tuple containing lists of fidelities and simulation times.
        """
        csocket: Socket = context.csockets[self.PEER_NAME]
        epr_socket: EPRSocket = context.epr_sockets[self.PEER_NAME]
        connection: BaseNetQASMConnection = context.connection

        for _ in range(self._num_epr_rounds):
            # Receive an EPR pair
            b1_qubit = epr_socket.recv_keep()[0]
            yield from connection.flush()

            # Create a local qubit to act as the target qubit for CNOT
            bob_qubit = Qubit(connection)

            # Receive Alice's measurements
            a1_measurement = yield from csocket.recv()
            alice_measurement = yield from csocket.recv()

            # Apply corrections based on Alice's measurements
            if a1_measurement == "1":
                b1_qubit.X()
            if alice_measurement == "1":
                b1_qubit.Z()

            # Perform CNOT operation and measure the qubit
            b1_qubit.cnot(bob_qubit)

            # Bob send back his state to Alice
            teleport_send(bob_qubit, context, self.PEER_NAME)

            b1_qubit.measure()
            yield from connection.flush()

            # Compute the fidelity of the final state
            state_ref = np.array([0, 1], dtype=complex)
            fidelity = compute_fidelity(bob_qubit, "Bob", state_ref)

            bob_qubit.measure()

            self.fidelities.append(fidelity)
            self.simulation_times.append(sim_time(MILLISECOND))

            yield from connection.flush()

        return self.fidelities, self.simulation_times
