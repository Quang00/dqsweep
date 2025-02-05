import logging
import math
from dataclasses import dataclass

import numpy
from netqasm.sdk import Qubit
from netqasm.sdk.toolbox.state_prep import set_qubit_state
from netsquid.qubits.dmutil import dm_fidelity

from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import get_qubit_state, get_reference_state
from squidasm.util.routines import teleport_recv, teleport_send
from squidasm.run.stack.config import StackNetworkConfig


@dataclass
class TeleportParams:
    phi: float = 0.0
    theta: float = 0.0

    @classmethod
    def generate_random_params(cls):
        params = cls()
        params.theta = numpy.random.random() * numpy.pi
        params.phi = numpy.random.random() * numpy.pi * 2
        return params

def extract_params_from_dm(dm):
    a = math.sqrt(dm[0, 0].real)
    b = math.sqrt(dm[1, 1].real)
    theta = 2 * math.acos(a)
    if a * b > 1e-12:
        phi = (-numpy.angle(dm[0, 1])) % (2 * math.pi)
    else:
        phi = 0.0
    return TeleportParams(phi=phi, theta=theta)

class AlicePingpongTeleportation(Program):
    PEER_NAME = "Bob"

    def __init__(self, num_epr_rounds):
        self._num_epr_rounds = num_epr_rounds
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)
        self.current_params = TeleportParams.generate_random_params()
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
        # In round 0, Alice sends her prepared qubit.
        for round in range(self._num_epr_rounds):
            if round % 2 == 0:
                # Even rounds: send the qubit.
                q = Qubit(context.connection)
                set_qubit_state(q, self.current_params.phi, self.current_params.theta)
                self.logger.info(f"Round {round}: Alice sending qubit with state: {self.current_params}")
                yield from teleport_send(q, context, peer_name=self.PEER_NAME)
            else:
                # Odd rounds: receive the qubit.
                q = yield from teleport_recv(context, peer_name=self.PEER_NAME)
                dm_received = get_qubit_state(q, "Alice")
                # Get the ideal (reference) density matrix from current parameters.
                dm_expected = get_reference_state(self.current_params.phi, self.current_params.theta)
                # Compute fidelity.
                fid = dm_fidelity(dm_expected, dm_received, squared=False, dm_check=True)
                self.fidelities.append(fid)
                self.logger.info(f"Round {round}: Alice received qubit, fidelity: {fid}")
                # Update the state parameters from the received state.
                self.current_params = extract_params_from_dm(dm_received)
        return self.fidelities


class BobPingpongTeleporation(Program):
    PEER_NAME = "Alice"

    def __init__(self, num_epr_rounds):
        self._num_epr_rounds = num_epr_rounds
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)
        self.current_params = None  # Will be updated upon reception.
        self.fidelities = []  # Record fidelities on reception rounds.

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="pingpong",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        for round in range(self._num_epr_rounds):
            if round % 2 == 0:
                # Even rounds: receive the qubit.
                q = yield from teleport_recv(context, peer_name=self.PEER_NAME)
                dm_received = get_qubit_state(q, "Bob")
                self.logger.info(f"Round {round}: Bob received qubit, density matrix:\n{dm_received}")
                # For Bob, we simply extract and store the state.
                self.current_params = extract_params_from_dm(dm_received)
                # Here, you could also compare with an expected state if you have one.
                # For demonstration, we assume Bob's expected state is the same as received.
                fid = dm_fidelity(dm_received, dm_received, squared=False, dm_check=True)
                self.fidelities.append(fid)
            else:
                # Odd rounds: send the qubit.
                if self.current_params is None:
                    self.logger.error("Bob has no state to send!")
                    break
                q = Qubit(context.connection)
                set_qubit_state(q, self.current_params.phi, self.current_params.theta)
                self.logger.info(f"Round {round}: Bob sending qubit with state: {self.current_params}")
                yield from teleport_send(q, context, peer_name=self.PEER_NAME)
        return  self.fidelities
    
if __name__ == "__main__":
    # Create a two-node network with nodes "Alice" and "Bob"
    cfg = StackNetworkConfig.from_file('configurations/perfect.yaml')

    num_epr_rounds = 2  # Total rounds. (Reception occurs in half of the rounds.)
    alice_program = AlicePingpongTeleportation(num_epr_rounds=num_epr_rounds)
    bob_program = BobPingpongTeleporation(num_epr_rounds=num_epr_rounds)

    # Set logger levels as needed.
    alice_program.logger.setLevel(logging.INFO)
    bob_program.logger.setLevel(logging.INFO)

    # Run the simulation.
    alice_result, bob_result = run(
        config=cfg,
        programs={"Alice": alice_program, "Bob": bob_program},
        num_times=10,
    )

    print(alice_result)
    print(bob_result)