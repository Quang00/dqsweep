from typing import Callable, Generator, List

from netqasm.sdk import Qubit

from squidasm.sim.stack.program import ProgramContext
from squidasm.util.routines import teleport_recv, teleport_send


# =============================================================================
# Routines
# =============================================================================
def pingpong_initiator(
    qubit: Qubit, context: ProgramContext, peer_name: str, num_rounds: int = 3
):
    """Executes the ping‐pong teleportation protocol for the initiator.

    In even rounds, the provided qubit is sent to the peer.
    In odd rounds, the initiator receives the qubit.
    The formal return is a generator and requires use of `yield from`
    in usage in order to function as intended.

    Args:
        qubit (Qubit): The qubit to be teleported.
        context (ProgramContext): Connection, csockets, and epr_sockets.
        peer_name (str): Name of the peer.
        num_rounds (int): Number of ping‐pong rounds.
    """
    if num_rounds % 2 == 0:
        raise ValueError("It must be odd for a complete ping-pong exchange.")

    for round_num in range(num_rounds):
        if round_num % 2 == 0:
            # Even round: send the qubit.
            yield from teleport_send(qubit, context, peer_name)
        else:
            # Odd round: receive a new qubit from the peer.
            qubit = yield from teleport_recv(context, peer_name)

    yield from context.connection.flush()


def pingpong_responder(
    context: ProgramContext, peer_name: str, num_rounds: int = 3
) -> Generator[None, None, Qubit]:
    """Executes the complementary ping‐pong teleportation protocol
    for the responder.

    The responder starts without a qubit and in the first (even) round
    receives one. In odd rounds he sends the qubit. After completing
    the rounds, Bob returns the final qubit he holds.
    The formal return is a generator and requires use of `yield from`
    in usage in order to function as intended.

    Args:
        context (ProgramContext): Connection, csockets, and epr_sockets.
        peer_name (str): Name of the peer.
        num_rounds (int): Number of ping‐pong rounds.

    Returns:
        Generator[None, None, Qubit]: The final teleported qubit.
    """
    if num_rounds % 2 == 0:
        raise ValueError("It must be odd for a complete ping-pong exchange.")

    qubit = None

    for round_num in range(num_rounds):
        if round_num % 2 == 0:
            # Even round: receive a qubit from the peer.
            qubit = yield from teleport_recv(context, peer_name)
        else:
            # Odd round: send the qubit to the peer.
            yield from teleport_send(qubit, context, peer_name)

    yield from context.connection.flush()

    return qubit


def distributed_n_qubit_controlled_u_control(
    context: ProgramContext, peer_name: str, ctrl_qubit: Qubit
) -> Generator[None, None, None]:
    """Performs the n-qubit controlled-U gate, but with one control qubit
    located on this node, the target on a remote node. The formal return is a
    generator and requires use of `yield from` in usage in order to function
    as intended.

    Args:
        context (ProgramContext): Context of the current program.
        peer_name (str): Name of the peer.
        ctrl_qubit (Qubit): The control qubit.
    """
    csocket = context.csockets[peer_name]
    epr_socket = context.epr_sockets[peer_name]
    connection = context.connection

    epr = epr_socket.create_keep()[0]
    ctrl_qubit.cnot(epr)
    epr_meas = epr.measure()
    yield from connection.flush()

    csocket.send(str(epr_meas))
    target_meas = yield from csocket.recv()
    if target_meas == "1":
        ctrl_qubit.Z()

    yield from connection.flush()


def distributed_n_qubit_controlled_u_target(
    context: ProgramContext,
    peer_names: List[str],
    target_qubit: Qubit,
    controlled_u: Callable[..., None],
):
    """Performs the n-qubit controlled-U gate, but with the target qubit
    located on this node, the controls on remote nodes. The formal return is
    a generator and requires use of `yield from` in usage in order to function
    as intended.

    Args:
        context (ProgramContext): Context of the current program.
        peer_names (List[str]): Name of the peers engaging.
        target_qubit (Qubit): The target qubit.
        controlled_u (Callable[..., None]): The n-qubits gate U with this
        signature `U(control_qubit_1, control_qubit_2, ..., target_qubit)`.
    """
    connection = context.connection

    epr_dict = {}
    for peer_name in peer_names:
        epr_dict[peer_name] = context.epr_sockets[peer_name].recv_keep()[0]
    yield from connection.flush()

    for peer_name, epr in epr_dict.items():
        m = yield from context.csockets[peer_name].recv()
        if m == "1":
            epr.X()

    epr_list = [epr_dict[peer_name] for peer_name in peer_names]
    controlled_u(*epr_list, target_qubit)

    epr_meas = {}
    for peer_name, epr in epr_dict.items():
        epr.H()
        epr_meas[peer_name] = epr.measure()
    yield from connection.flush()

    for peer_name in peer_names:
        context.csockets[peer_name].send(str(epr_meas[peer_name]))
