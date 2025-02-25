"""

"""

import numpy as np

from netqasm.sdk.qubit import Qubit

from netsquid.qubits.dmutil import dm_fidelity
from netsquid.util.simtools import MILLISECOND, sim_time

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import get_qubit_state
from squidasm.util.routines import teleport_recv, teleport_send



# =============================================================================
# Alice's Program for grover
# =============================================================================
class AliceGrover(Program):
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
            name="grover",
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

        for _ in range(self._num_epr_rounds):
            alice_1 = Qubit(context.connection)
            alice_1.X()
            yield from teleport_send(alice_1, context, peer_name=self.PEER_NAME)
            alice_2 = Qubit(context.connection)
            alice_2.X()
            yield from teleport_send(alice_2, context, peer_name=self.PEER_NAME)
            yield from context.connection.flush()

        return {}


# =============================================================================
# Bob's Program for Nonlocal CNOT
# =============================================================================
class BobGrover(Program):
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
            name="grover",
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

        fidelities: list[float] = []
        simulation_times: list[float] = []

        for _ in range(self._num_epr_rounds):
           
            # Receive EPR pair from Al
            bob_qubit = Qubit(context.connection)
            bob_qubit.X()
            oracle = yield from teleport_recv(context, peer_name=self.PEER_NAME)
            oracle.cnot(bob_qubit)

            alice = yield from teleport_recv(context, peer_name=self.PEER_NAME)
            alice.cnot(bob_qubit)

            yield from context.connection.flush()
            # Compute fidelity of the final state
            dm_b = get_qubit_state(bob_qubit, "Bob", full_state=True)
            state_ref = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=complex)
            dm_ref = np.outer(state_ref, np.conjugate(state_ref))
            fidelity = dm_fidelity(dm_b, dm_ref)
            fidelities.append(fidelity)

            # Free qubit and record simulation time
            bob_qubit.free()
            oracle.free()
            alice.free()
            yield from context.connection.flush()
            simulation_times.append(sim_time(MILLISECOND))

        return fidelities, simulation_times
