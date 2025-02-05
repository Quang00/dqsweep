import logging
import math
import numpy as np

from netqasm.sdk import Qubit
from netqasm.sdk.toolbox.state_prep import set_qubit_state
from netsquid.qubits.dmutil import dm_fidelity
from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import get_qubit_state, get_reference_state
from squidasm.util.routines import teleport_recv, teleport_send

# Fixed state for the pingpong teleportation protocol.
FIXED_THETA = np.pi / 2
FIXED_PHI = 0.0

def extract_params_from_dm(dm):
    """
    Extracts phi and theta from a 2x2 density matrix of a pure state.

    Assumes the density matrix is given by:
      [[cos^2(theta/2), cos(theta/2)*exp(-i*phi)*sin(theta/2)],
       [cos(theta/2)*exp(i*phi)*sin(theta/2), sin^2(theta/2)]]
    """
    a = math.sqrt(dm[0, 0].real)  # ~ cos(theta/2)
    b = math.sqrt(dm[1, 1].real)  # ~ sin(theta/2)
    theta = 2 * math.acos(a)
    if a * b > 1e-12:
        phi = (-np.angle(dm[0, 1])) % (2 * math.pi)
    else:
        phi = 0.0
    return phi, theta

class AlicePingpongTeleportation(Program):
    PEER_NAME = "Bob"

    def __init__(self, num_epr_rounds):
        self._num_epr_rounds = num_epr_rounds
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)
        self.current_phi = FIXED_PHI
        self.current_theta = FIXED_THETA
        self.fidelities = []

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="pingpong",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        self.fidelities = []
        # For a total of num_epr_rounds, alternate sending (even rounds)
        # and receiving (odd rounds).
        for round in range(self._num_epr_rounds):
            if round % 2 == 0:
                # Even rounds: send the qubit.
                q = Qubit(context.connection)
                set_qubit_state(q, self.current_phi, self.current_theta)
                self.logger.info(
                    f"Round {round}: Alice sending qubit with state: phi={self.current_phi}, theta={self.current_theta}"
                )
                yield from teleport_send(q, context, peer_name=self.PEER_NAME)
            else:
                # Odd rounds: receive the qubit.
                q = yield from teleport_recv(context, peer_name=self.PEER_NAME)
                dm_received = get_qubit_state(q, "Alice")
                dm_expected = get_reference_state(FIXED_PHI, FIXED_THETA)
                fid = dm_fidelity(dm_received, dm_expected)
                self.fidelities.append(fid)
                self.logger.info(
                    f"Round {round}: Alice received qubit with fidelity: {fid}"
                )
                # Update current state from the received qubit (even though under ideal conditions it should remain the same)
                self.current_phi, self.current_theta = extract_params_from_dm(dm_received)
        return self.fidelities

class BobPingpongTeleporation(Program):
    PEER_NAME = "Alice"

    def __init__(self, num_epr_rounds):
        self._num_epr_rounds = num_epr_rounds
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)
        self.current_phi = None  # Will be updated upon reception.
        self.current_theta = None
        self.fidelities = []

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="pingpong",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        self.fidelities = []
        # For a total of num_epr_rounds, alternate between receiving (even rounds)
        # and sending (odd rounds).
        for round in range(self._num_epr_rounds):
            if round % 2 == 0:
                # Even rounds: receive the qubit.
                q = yield from teleport_recv(context, peer_name=self.PEER_NAME)
                dm_received = get_qubit_state(q, "Bob")
                self.current_phi, self.current_theta = extract_params_from_dm(dm_received)
                dm_expected = get_reference_state(FIXED_PHI, FIXED_THETA)
                fid = dm_fidelity(dm_received, dm_expected)
                self.fidelities.append(fid)
                self.logger.info(f"Round {round}: Bob received qubit with fidelity: {fid}")
            else:
                # Odd rounds: send the qubit.
                if self.current_phi is None or self.current_theta is None:
                    self.logger.error("Bob has no state to send!")
                    break
                q = Qubit(context.connection)
                set_qubit_state(q, self.current_phi, self.current_theta)
                self.logger.info(
                    f"Round {round}: Bob sending qubit with state: phi={self.current_phi}, theta={self.current_theta}"
                )
                yield from teleport_send(q, context, peer_name=self.PEER_NAME)
        return self.fidelities

if __name__ == "__main__":
    cfg = StackNetworkConfig.from_file('configurations/generic_qdevice.yaml')
    num_epr_rounds = 100
    num_experiments = 1

    alice_program = AlicePingpongTeleportation(num_epr_rounds=num_epr_rounds)
    bob_program = BobPingpongTeleporation(num_epr_rounds=num_epr_rounds)

    alice_program.logger.setLevel(logging.INFO)
    bob_program.logger.setLevel(logging.INFO)

    alice_results, bob_results = run(
        config=cfg,
        programs={"Alice": alice_program, "Bob": bob_program},
        num_times=num_experiments,
    )

    avg_alice_fidelities = np.mean(alice_results) * 100
    avg_bob_fidelities = np.mean(bob_results) * 100

    print("Alice fidelities:", avg_alice_fidelities)
    print("Bob fidelities:", avg_bob_fidelities)
