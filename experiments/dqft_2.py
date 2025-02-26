"""
DQFT2 Quantum Experiment
--------------------------------

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
# Alice's Program for DQFT
# =============================================================================
class AliceDQFT2(Program):
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
            name="dqft",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
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
            a1_qubit = epr_socket.recv_keep()[0]
            
            yield from connection.flush()
            b1_measurement = yield from csocket.recv()
            if b1_measurement == "1":
                a1_qubit.X()

            # Create a local qubit and apply an X gate
            alice_qubit = Qubit(connection)
            alice_qubit.H()

            alice_qubit.T()

            alice_qubit.cnot(a1_qubit)
            a1_qubit.Z()
            a1_qubit.S()
            a1_qubit.T()
            alice_qubit.cnot(a1_qubit)
            a1_qubit.T()
            a1_qubit.H()
            a1_measurement = a1_qubit.measure()

            #yield from connection.flush()
            #csocket.send(str(a1_measurement))
            print("-----------------------------")

            # Free qubit and flush the connection

            yield from connection.flush()

        return {}


# =============================================================================
# Bob's Program for Nonlocal CNOT
# =============================================================================
class BobDQFT2(Program):
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
            name="dqft",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
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
            b1_qubit = epr_socket.create_keep()[0]

            # Create a local qubit as the target of the CNOT operation
            bob_qubit = Qubit(connection)

            bob_qubit.cnot(b1_qubit)
            b1_measurement = b1_qubit.measure()
            yield from connection.flush()
            csocket.send(str(b1_measurement))
            '''
            yield from connection.flush()
            a1_measurement = yield from csocket.recv()
            if a1_measurement == "1":
                bob_qubit.Z()
            bob_qubit.H()
            '''

            yield from connection.flush()

            # Compute fidelity of the final state
            dm_b = get_qubit_state(bob_qubit, "Bob", full_state=True)
            print(dm_b)
            state_ref = np.array([1, 0, 0, 0], dtype=complex)
            dm_ref = np.outer(state_ref, np.conjugate(state_ref))
            fidelity = dm_fidelity(dm_b, dm_ref)
            print(fidelity)
            fidelities.append(fidelity)

            # Free qubit and record simulation time
            bob_qubit.free()
            yield from connection.flush()
            simulation_times.append(sim_time(MILLISECOND))
        print(fidelities)
        return fidelities, simulation_times
