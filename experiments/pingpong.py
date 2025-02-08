"""
Ping-Pong Teleportation Experiment
-----------------------------------

This module implements a quantum teleportation ping-pong experiment
between two nodes (Alice and Bob). The experiment involves alternating
qubit transmission and reception, with fidelity checks against a
reference state. This is only for experimental purposes since you cannot
extract the state of a qubit in a real-world scenario.

Experiment details:
  - Alice sends a fixed-state qubit during even rounds and receives one
    during odd rounds.
  - Bob does the opposite: receiving during even rounds and sending
    during odd rounds.
  - Upon reception, the receiver extracts state parameters and computes
    fidelity against the reference state.
"""

import numpy as np
from netqasm.sdk import Qubit
from netqasm.sdk.toolbox.state_prep import set_qubit_state
from netsquid.qubits.dmutil import dm_fidelity
from netsquid.util.simtools import MILLISECOND, sim_time
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import get_qubit_state, get_reference_state
from squidasm.util.routines import teleport_recv, teleport_send

# =============================================================================
# Constants: Fixed Qubit State Parameters
# =============================================================================
THETA = 0.0
PHI = 0.0


def extract_params_from_dm(dm):
    """
    Extract phase (phi) and amplitude (theta) from a density matrix.

    Args:
        dm (ndarray): 2x2 density matrix.

    Returns:
        tuple: (phi, theta) extracted values.
    """
    a = np.clip(np.sqrt(np.real(dm[0, 0])), 0.0, 1.0)
    theta = 2 * np.arccos(a)
    phi = (-np.angle(dm[0, 1]) if abs(dm[0, 1]) > 1e-12 else 0.0) % (2 * np.pi)
    return phi, theta


# =============================================================================
# Alice's Ping-Pong Teleportation Program
# =============================================================================
class AlicePingpongTeleportation(Program):
    """Implements Alice's side of the ping-pong teleportation experiment.

    Alice alternates between sending and receiving qubits across multiple
    rounds. Even rounds involve sending a qubit prepared in a predefined
    state, while odd rounds involve receiving a qubit and computing the
    fidelity against an expected reference state.

    Args:
        num_epr_rounds (int): Number of EPR rounds for the experiment.
    """

    PEER_NAME = "Bob"

    def __init__(self, num_epr_rounds: int):
        """Initializes Alice's program with the specified number of rounds.

        Args:
            num_epr_rounds (int): Number of EPR rounds in the experiment.
        """
        super().__init__()
        self._num_epr_rounds = num_epr_rounds
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)
        self.current_phi: float = PHI
        self.current_theta: float = THETA
        self.fidelities: list[float] = []
        self.simulation_times: list[float] = []

    @property
    def meta(self) -> ProgramMeta:
        """Defines metadata for Alice's teleportation program.

        Returns:
            ProgramMeta: Metadata including experiment name, sockets, and qubit limit.
        """
        return ProgramMeta(
            name="pingpong",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        """Executes Alice's teleportation routine with alternating rounds.

        In even rounds, Alice prepares and sends a qubit. In odd rounds,
        she receives a qubit, extracts its state, and computes the fidelity
        against a reference state.

        Args:
            context (ProgramContext): Provides network and connection details.

        Returns:
            tuple[list[float], list[float]]: Lists containing fidelities and
            simulation times for each round.
        """
        for epr_round in range(self._num_epr_rounds):
            if epr_round % 2 == 0:
                # Even rounds: Send a qubit
                qubit = Qubit(context.connection)
                set_qubit_state(qubit, self.current_phi, self.current_theta)
                self.logger.info(f"Alice Round {epr_round}: Sending qubit")
                yield from teleport_send(qubit, context, peer_name=self.PEER_NAME)
            else:
                # Odd rounds: Receive a qubit and compute fidelity
                qubit = yield from teleport_recv(context, peer_name=self.PEER_NAME)
                dm_received = get_qubit_state(qubit, "Alice")
                dm_expected = get_reference_state(PHI, THETA)
                self.current_phi, self.current_theta = extract_params_from_dm(
                    dm_received
                )
                fid = dm_fidelity(dm_received, dm_expected, dm_check=False)
                self.fidelities.append(fid)
                self.logger.info(f"Alice Round {epr_round}: Fidelity = {fid}")
                self.simulation_times.append(sim_time(MILLISECOND))

        return self.fidelities, self.simulation_times


# =============================================================================
# Bob's Ping-Pong Teleportation Program
# =============================================================================
class BobPingpongTeleportation(Program):
    """Implements Bob's side of the ping-pong teleportation experiment.

    Bob alternates between receiving and sending qubits across multiple
    rounds. Even rounds involve receiving a qubit and extracting its state,
    while odd rounds involve sending a qubit prepared using the extracted state.

    Args:
        num_epr_rounds (int): Number of EPR rounds for the experiment.
    """

    PEER_NAME = "Alice"

    def __init__(self, num_epr_rounds: int):
        """Initializes Bob's program with the specified number of rounds.

        Args:
            num_epr_rounds (int): Number of EPR rounds in the experiment.
        """
        super().__init__()
        self._num_epr_rounds = num_epr_rounds
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)
        self.current_phi: float | None = None
        self.current_theta: float | None = None
        self.fidelities: list[float] = []
        self.simulation_times: list[float] = []

    @property
    def meta(self) -> ProgramMeta:
        """Defines metadata for Bob's teleportation program.

        Returns:
            ProgramMeta: Metadata including experiment name, sockets, and qubit limit.
        """
        return ProgramMeta(
            name="pingpong",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        """Executes Bob's teleportation routine with alternating rounds.

        In even rounds, Bob receives a qubit, extracts its state, and computes
        the fidelity against a reference state. In odd rounds, he sends a qubit
        using the previously extracted state parameters.

        Args:
            context (ProgramContext): Provides network and connection details.

        Returns:
            tuple[list[float], list[float]]: Lists containing fidelities and
            simulation times for each round.
        """
        for epr_round in range(self._num_epr_rounds):
            if epr_round % 2 == 0:
                # Even rounds: Receive a qubit and extract its state
                qubit = yield from teleport_recv(context, peer_name=self.PEER_NAME)
                dm_received = get_qubit_state(qubit, "Bob")
                self.current_phi, self.current_theta = extract_params_from_dm(
                    dm_received
                )
                dm_expected = get_reference_state(PHI, THETA)
                fid = dm_fidelity(dm_received, dm_expected, dm_check=False)
                self.fidelities.append(fid)
                self.logger.info(f"Bob Round {epr_round}: Fidelity = {fid}")
                self.simulation_times.append(sim_time(MILLISECOND))
            else:
                # Odd rounds: Send a qubit using the extracted state
                if self.current_phi is None or self.current_theta is None:
                    self.logger.error("Bob has no state to send!")
                    break
                qubit = Qubit(context.connection)
                set_qubit_state(qubit, self.current_phi, self.current_theta)
                self.logger.info(f"Bob Round {epr_round}: Sending qubit")
                yield from teleport_send(qubit, context, peer_name=self.PEER_NAME)

        return self.fidelities, self.simulation_times
