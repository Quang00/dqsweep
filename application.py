from netqasm.sdk.classical_communication.socket import Socket
from netqasm.sdk.connection import BaseNetQASMConnection
from netqasm.sdk.epr_socket import EPRSocket
from netqasm.sdk.qubit import Qubit
from netsquid.qubits.dmutil import dm_fidelity

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import get_qubit_state

import numpy as np

class AliceProgram(Program):
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
            alice_qubit.reset()
            alice_qubit.X()

            # Perfom a CNOT gate between alice qubit and her shared entangled qubit a1
            alice_qubit.cnot(a1_qubit)
            
            # Alice measures her entangled qubit a1
            a1_measurement = a1_qubit.measure()
            yield from connection.flush()

            # Alice send her measurement to Bob on a classical channel
            csocket.send(str(a1_measurement))

            # Alice listens the classical channel to get Bob's measurement
            b1_measurement = yield from csocket.recv()
            if b1_measurement == '1':
                alice_qubit.Z()

            alice_qubit.free()
            yield from connection.flush()

        return {}


class BobProgram(Program):
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
        for _ in range(self._num_epr_rounds):
            # Listen for request to create EPR pair
            b1_qubit = epr_socket.recv_keep()[0]

            yield from connection.flush()

            # Create a local qubit which is the target qubit of the distributed CNOT gate
            bob_qubit = Qubit(connection)
            bob_qubit.reset()
        
            # Bob listens the classical channel to get the measurement from Alice
            a1_measurement = yield from csocket.recv()

            # At this stage a distributed CNOT was perfomed between Alice qubit and b1 qubit
            if a1_measurement == '1':
                b1_qubit.X()
            
            b1_qubit.cnot(bob_qubit)
            b1_qubit.H()
            b1_measurement = b1_qubit.measure()

            yield from connection.flush()

            # Bob send the measurement to Alice on a classical channel
            csocket.send(str(b1_measurement))
            dm_b = get_qubit_state(bob_qubit, "Bob", full_state=True)

            # Create the perfect resulting density matrix to assess the fidelity
            state_ref = np.array([0, 0, 0, 1], dtype=complex)
            dm_ref = np.outer(state_ref, np.conjugate(state_ref))
        
            # Compute the fidelity between the two density matrices    
            fidelity = dm_fidelity(dm_b, dm_ref)
            fidelities.append(fidelity)

            bob_qubit.free()
            yield from connection.flush()

        return fidelities
