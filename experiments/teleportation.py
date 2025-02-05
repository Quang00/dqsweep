import numpy as np
from netqasm.sdk.classical_communication.socket import Socket
from netqasm.sdk.connection import BaseNetQASMConnection
from netqasm.sdk.epr_socket import EPRSocket
from netqasm.sdk.qubit import Qubit
from netsquid.qubits.dmutil import dm_fidelity
from netsquid.util.simtools import sim_time, MILLISECOND
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import get_qubit_state


class AliceTeleportation(Program):
    PEER_NAME = "Bob"

    def __init__(self, num_epr_rounds):
        self._num_epr_rounds = num_epr_rounds

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="nonlocal_CNOT",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=1,
        )

    def run(self, context: ProgramContext):
        # get classical socket to peer
        csocket = context.csockets[self.PEER_NAME]
        # get EPR socket to peer
        epr_socket = context.epr_sockets[self.PEER_NAME]
        # get connection to quantum network processing unit
        connection = context.connection

        for _ in range(self._num_epr_rounds):
            # Register a request to create an EPR pair
            a1_qubit = epr_socket.create_keep()[0]

            # Create a local qubit which is the control qubit of the distributed CNOT gate
            alice_qubit = Qubit(connection)
            alice_qubit.X()

            # Perfom a CNOT gate between alice qubit and her shared entangled qubit a1
            alice_qubit.cnot(a1_qubit)
            alice_qubit.H()

            # Alice measures her two qubits
            a1_measurement = a1_qubit.measure()
            alice_measurement = alice_qubit.measure()
            yield from connection.flush()

            # Alice send her measurements to Bob on a classical channel
            csocket.send(str(a1_measurement))
            csocket.send(str(alice_measurement))
            yield from connection.flush()

        return {}


class BobTeleportation(Program):
    PEER_NAME = "Alice"

    def __init__(self, num_epr_rounds):
        self._num_epr_rounds = num_epr_rounds

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="nonlocal_CNOT",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=1,
        )

    def run(self, context: ProgramContext):
        # get classical socket to peer
        csocket: Socket = context.csockets[self.PEER_NAME]
        # get EPR socket to peer
        epr_socket: EPRSocket = context.epr_sockets[self.PEER_NAME]
        # get connection to quantum network processing unit
        connection: BaseNetQASMConnection = context.connection

        fidelities = []
        simulation_times = []
        for _ in range(self._num_epr_rounds):
            # Listen for request to create EPR pair
            b1_qubit = epr_socket.recv_keep()[0]

            yield from connection.flush()

            # Create a local qubit which is the target qubit of the distributed CNOT gate
            bob_qubit = Qubit(connection)

            # Bob listens the classical channel to get the measurements from Alice
            a1_measurement = yield from csocket.recv()
            alice_measurement = yield from csocket.recv()

            if a1_measurement == "1":
                b1_qubit.X()
            if alice_measurement == "1":
                b1_qubit.Z()

            b1_qubit.cnot(bob_qubit)
            b1_qubit.measure()
            yield from connection.flush()

            dm_b = get_qubit_state(bob_qubit, "Bob", full_state=True)
 
            # Create the perfect resulting density matrix to assess the fidelity
            state_ref = np.array([0, 1], dtype=complex)
            dm_ref = np.outer(state_ref, np.conjugate(state_ref))

            # Compute the fidelity between the two density matrices
            fidelity = dm_fidelity(dm_b, dm_ref)
            fidelities.append(fidelity)

            bob_qubit.free()
            yield from connection.flush()
            simulation_times.append(sim_time(MILLISECOND))

        return fidelities, simulation_times
