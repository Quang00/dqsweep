"""
Nonlocal CNOT Gate With Two Teleportations
------------------------------------------

This module implements a nonlocal CNOT gate between two parties:
Alice and Bob using two ebits and two bits in each direction. The
implementation is a general implementation of any nonlocal two qubit
gate, which uses two quantum teleportations. This approach consumes more
resources that the one implemented in the file `nonlocal_cnot.py`.

The initial control qubit state of Alice is |1>, the initial target
qubit state of Bob is |0>. After the distributed CNOT gate, the qubit
of Bob should be |1>. This is verified by computing the fidelity
between the density output matrix and the expected one.

Note: The first teleportation was implemented manually with the whole protocol
while the second teleportation uses the provided routines by squidasm.
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
# Alice's Program for Nonlocal CNOT Gate using two teleportations
# =============================================================================
class Alice2Teleportations(Program):
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
            name="teleportation",
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
            # Create an EPR pair for the first teleportation
            a1_qubit = epr_socket.create_keep()[0]

            # Create a local qubit which is the control qubit of the CNOT gate
            alice_qubit = Qubit(connection)
            alice_qubit.X()
            alice_qubit.cnot(a1_qubit)
            alice_qubit.H()

            # Measure both qubits
            a1_measurement = a1_qubit.measure()
            alice_measurement = alice_qubit.measure()
            yield from connection.flush()

            # First teleportation is complete after the 2 measures send to Bob
            csocket.send(str(a1_measurement))
            csocket.send(str(alice_measurement))

            # Alice receives the state of Bob after the CNOT gate was applied
            qubit = yield from teleport_recv(context, self.PEER_NAME)
            qubit.measure()
            yield from connection.flush()

        return {}


# =============================================================================
# Bob's for Nonlocal CNOT Gate using two teleportations
# =============================================================================
class Bob2Teleportations(Program):
    """Implements Bob's side of the nonlocal CNOT gate.
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
        """
        Defines metadata for Bob's program.

        Returns:
            ProgramMeta: Experiment name, sockets, qubit limit.
        """
        return ProgramMeta(
            name="teleportation",
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

        for _ in range(self._num_epr_rounds):
            # Receive an EPR pair for the first teleportation
            b1_qubit = epr_socket.recv_keep()[0]
            yield from connection.flush()

            # Create a local qubit which is the target qubit of the CNOT gate
            bob_qubit = Qubit(connection)

            # Receive Alice's measurements
            a1_measurement = yield from csocket.recv()
            alice_measurement = yield from csocket.recv()

            # Apply corrections based on Alice's measurements
            if a1_measurement == "1":
                b1_qubit.X()
            if alice_measurement == "1":
                b1_qubit.Z()

            # Perform the distributed CNOT gate locally since Bob has now the
            # state of Alice in his qubit b1. And then swap the qubit with the
            # equivalent set of gates (1 SWAP = 3 CNOT) so we can only use 2
            # CNOT gates to optimize the circuit.
            bob_qubit.cnot(b1_qubit)
            b1_qubit.cnot(bob_qubit)

            b1_qubit.measure()
            yield from connection.flush()

            # Compare density matrices with the expected state
            state_ref = np.array([0, 1], dtype=complex)
            fidelity = compute_fidelity(bob_qubit, "Bob", state_ref)

            # Bob sends back his state to Alice through a second teleportation
            yield from teleport_send(bob_qubit, context, self.PEER_NAME)

            # Store the fidelity and simulation time results
            self.fidelities.append(fidelity)
            self.simulation_times.append(sim_time(MILLISECOND))

            yield from connection.flush()

        return self.fidelities, self.simulation_times
