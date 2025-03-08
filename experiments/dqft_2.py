"""
DQFT2 Quantum Experiment
--------------------------------
This file implements a distributed quantum Fourier transform (QFT) on 2 qubits,
between two nodes (Alice and Bob).
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
# Alice's Program for DQFT2
# =============================================================================
class AliceDQFT2(Program):
    """
    Implements Alice's side of the distributed QFT experiment.

    Args:
        num_epr_rounds (int):  Number of EPR rounds for the experiment.
    """

    PEER_NAME = "Bob"

    def __init__(self, num_epr_rounds: int):
        """Initializes Alice's program with the specified number of rounds."""
        self._num_epr_rounds = num_epr_rounds

    @property
    def meta(self) -> ProgramMeta:
        """Defines metadata for Alice's DQFT2 program.

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
        Executes Alice's part of the DQFT2.

        """
        csocket = context.csockets[self.PEER_NAME]
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        for _ in range(self._num_epr_rounds):
            a1_qubit = epr_socket.recv_keep()[0]
            yield from connection.flush()

            b1_measurement = yield from csocket.recv()
            if b1_measurement == "1":
                a1_qubit.X()

            alice_qubit = Qubit(connection)
            alice_qubit.H()
            alice_qubit.rot_Z(angle=np.pi / 4)

            alice_qubit.cnot(a1_qubit)
            a1_qubit.rot_Z(angle=-np.pi / 4)
            alice_qubit.cnot(a1_qubit)

            a1_qubit.rot_Z(angle=np.pi / 4)
            a1_qubit.H()
            a1_measurement = a1_qubit.measure()
            yield from connection.flush()
            csocket.send(str(a1_measurement))

            alice_qubit.free()
            yield from connection.flush()

        return {}


# =============================================================================
# Bob's Program for DQFT2
# =============================================================================
class BobDQFT2(Program):
    """
    Implements Bob's side of the distributed QFT experiment.

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
        """Defines metadata for Bob's DQFT2 program.

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
        Executes Bob's part of the DQFT2.

        Returns:
            tuple[list[float], list[float]]: The fidelity and simulation time
            lists for each round.
        """
        csocket: Socket = context.csockets[self.PEER_NAME]
        epr_socket: EPRSocket = context.epr_sockets[self.PEER_NAME]
        connection: BaseNetQASMConnection = context.connection

        for _ in range(self._num_epr_rounds):
            b1_qubit = epr_socket.create_keep()[0]
            bob_qubit = Qubit(connection)

            bob_qubit.cnot(b1_qubit)
            b1_measurement = b1_qubit.measure()
            yield from connection.flush()
            csocket.send(str(b1_measurement))

            a1_measurement = yield from csocket.recv()
            if a1_measurement == "1":
                bob_qubit.Z()
            bob_qubit.H()

            yield from connection.flush()

            state_ref = np.array([1, 1, 1, 1], dtype=complex) * 0.5
            fidelity = compute_fidelity(bob_qubit, "Bob", state_ref)

            bob_qubit.measure()

            self.fidelities.append(fidelity)
            self.simulation_times.append(sim_time(MILLISECOND))

            yield from connection.flush()

        return self.fidelities, self.simulation_times
