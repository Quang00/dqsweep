"""
Nonlocal Toffoli Gate
------------------

This module implements a nonlocal Toffoli gate between three parties:
Alice, Bob and Charlie using two ebits and four bits in each direction. The
implementation is from the paper "Optimal local implementation
of nonlocal quantum gates" (2000), Eisert, Jens et al.

The initial control qubit state of Alice is |1>, the control qubit state of Bob
is |1>, the initial target qubit state of Charlie is |0>. After the distributed
Toffoli gate, the qubit of Charlie should be |1>. This is verified by computing
the fidelity between the density output matrix and the expected one.
"""

import numpy as np
from netqasm.sdk.qubit import Qubit
from netsquid.util.simtools import MILLISECOND, sim_time

from experiments.utils import (
    compute_fidelity,
    distributed_n_qubit_controlled_u_control,
    distributed_n_qubit_controlled_u_target,
    toffoli,
)
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta


class AliceToffoli(Program):
    PEER_NAME = "Charlie"

    def __init__(self, num_epr_rounds: int):
        """Initializes Alice's program with the given number of rounds.

        Args:
            num_epr_rounds (int): Number of EPR rounds.
        """
        self._num_epr_rounds = num_epr_rounds

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="nonlocal_Toffoli",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=3,
        )

    def run(self, context: ProgramContext):
        connection = context.connection

        for _ in range(self._num_epr_rounds):
            a_q = Qubit(connection)
            a_q.X()
            yield from distributed_n_qubit_controlled_u_control(
                context, self.PEER_NAME, a_q
            )
            a_q.measure()
            yield from connection.flush()

        return {}


class BobToffoli(Program):
    PEER_NAME = "Charlie"

    def __init__(self, num_epr_rounds: int):
        """Initializes Bob's program with the given number of rounds.

        Args:
            num_epr_rounds (int): Number of EPR rounds.
        """
        self._num_epr_rounds = num_epr_rounds

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="nonlocal_Toffoli",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=3,
        )

    def run(self, context: ProgramContext):
        connection = context.connection

        for _ in range(self._num_epr_rounds):
            b_q = Qubit(connection)
            b_q.X()
            yield from distributed_n_qubit_controlled_u_control(
                context, self.PEER_NAME, b_q
            )
            b_q.measure()
            yield from connection.flush()


class CharlieToffoli(Program):
    PEERS = ["Alice", "Bob"]

    def __init__(self, num_epr_rounds: int):
        """Initializes Charlie's program with the given number of rounds.

        Args:
            num_epr_rounds (int): Number of EPR rounds.
        """
        self._num_epr_rounds = num_epr_rounds

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="nonlocal_Toffoli",
            csockets=self.PEERS,
            epr_sockets=self.PEERS,
            max_qubits=3,
        )

    def run(self, context: ProgramContext):
        connection = context.connection

        fidelities: list[float] = []
        simulation_times: list[float] = []

        for _ in range(self._num_epr_rounds):
            c_q = Qubit(context.connection)
            yield from distributed_n_qubit_controlled_u_target(
                context, self.PEERS, c_q, toffoli
            )
            yield from connection.flush()

            state_ref = np.array([0, 1], dtype=complex)
            fidelity = compute_fidelity(c_q, "Charlie", state_ref, False)

            c_q.measure()

            # Store the fidelity and simulation time results
            fidelities.append(fidelity)
            simulation_times.append(sim_time(MILLISECOND))

            yield from connection.flush()

        return fidelities, simulation_times
