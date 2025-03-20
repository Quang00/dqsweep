"""
Pingpong Teleportation
----------------------

This module implements a quantum teleportation ping-pong experiment
between two parties: Alice and Bob. The experiment initializes a qubit
in a random state and then does a back-and-forth with it between the
two parties.

Alice sends this initial qubit during even rounds and receives the same
qubit during odd rounds. Bob does the opposite, he receives this qubit
during even rounds and sends it back to Alice during odd rounds.
"""

import numpy as np
from netqasm.sdk import Qubit
from netqasm.sdk.toolbox.state_prep import set_qubit_state
from netsquid.qubits.dmutil import dm_fidelity
from netsquid.util.simtools import MILLISECOND, sim_time

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import get_qubit_state, get_reference_state

from utils.routines import pingpong_initiator, pingpong_responder


# =============================================================================
# Initialize a random qubit state
# =============================================================================
rng = np.random.default_rng(42)
PHI = rng.random() * np.pi
THETA = rng.random() * np.pi


# =============================================================================
# Alice's Ping-Pong Teleportation Program
# =============================================================================
class AlicePingpongTeleportation(Program):
    """Implements Alice's side of the ping-pong teleportation experiment.

    Alice alternates between sending and receiving qubits across multiple
    rounds. In even rounds, Alice sends the qubit, while odd rounds Alice
    receives the qubit.
    """

    PEER_NAME = "Bob"

    def __init__(self, num_epr_rounds: int):
        """Initializes Alice's program with the given number of rounds.

        Args:
            num_epr_rounds (int): Number of EPR rounds.
        """
        self._num_epr_rounds = num_epr_rounds
        self.initial_phi: float = PHI
        self.initial_theta: float = THETA

    @property
    def meta(self) -> ProgramMeta:
        """Defines metadata for Alice's program.

        Returns:
            ProgramMeta: Experiment name, sockets, qubit limit.
        """
        return ProgramMeta(
            name="pingpong",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        """Executes Alice's routine.

        In even rounds, Alice prepares and sends a qubit. In odd rounds,
        she receives the qubit from Bob.

        Args:
            context (ProgramContext): Network and connection details.

        """
        qubit = Qubit(context.connection)
        set_qubit_state(qubit, self.initial_phi, self.initial_theta)

        qubit = yield from pingpong_initiator(
            qubit, context, self.PEER_NAME, self._num_epr_rounds
        )

        yield from context.connection.flush()


# =============================================================================
# Bob's Ping-Pong Teleportation Program
# =============================================================================
class BobPingpongTeleportation(Program):
    """Implements Bob's side of the ping-pong teleportation experiment.

    Bob alternates between receiving and sending qubits across multiple
    rounds. Even rounds involve receiving a qubit, while odd rounds involve
    sending the qubit.
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
            name="pingpong",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        """Executes Bob's routine.

        In even rounds, Bob receives a qubit.
        In odd rounds, Bob sends the qubit.

        Args:
            context (ProgramContext): Network and connection details.

        Returns:
            list[tuple[list[float], list[float]]]: A list of tuple containing
            lists of fidelities and simulation times.
        """
        fidelities = []
        simulation_times = []

        qubit = yield from pingpong_responder(
            context, self.PEER_NAME, self._num_epr_rounds
        )

        yield from context.connection.flush()

        # Compare density matrices with the expected state
        dm_received = get_qubit_state(qubit, "Bob")
        dm_expected = get_reference_state(PHI, THETA)
        fid = dm_fidelity(dm_received, dm_expected, dm_check=False)

        qubit.measure()

        fidelities.append(fid)
        simulation_times.append(sim_time(MILLISECOND))

        yield from context.connection.flush()

        return fidelities, simulation_times
