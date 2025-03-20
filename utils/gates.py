from typing import Callable, List

import numpy as np
from netqasm.sdk import Qubit

from squidasm.sim.stack.program import ProgramContext


# =============================================================================
# Gates
# =============================================================================
def toffoli(control1: Qubit, control2: Qubit, target: Qubit) -> None:
    """Performs a Toffoli gate with `control1` and `control2` as control qubits
    and `target` as target, using CNOT, Rz and Hadamard gates.

    See https://en.wikipedia.org/wiki/Toffoli_gate

    Args:
        control1 (Qubit): First control qubit.
        control2 (Qubit): Second control qubit.
        target (Qubit): Target qubit.
    """
    target.H()
    control2.cnot(target)
    target.rot_Z(angle=-np.pi / 4)
    control1.cnot(target)
    target.rot_Z(angle=np.pi / 4)
    control2.cnot(target)
    target.rot_Z(angle=-np.pi / 4)
    control1.cnot(target)
    control2.rot_Z(angle=np.pi / 4)
    target.rot_Z(angle=np.pi / 4)
    target.H()
    control1.cnot(control2)
    control1.rot_Z(angle=np.pi / 4)
    control2.rot_Z(angle=-np.pi / 4)
    control1.cnot(control2)


def ccz(control1: Qubit, control2: Qubit, target: Qubit) -> None:
    """Performs a CCZ gate with `control1` and `control2` as control qubits
    and `target` as target, using Toffoli and Hadamard gates.

    Args:
        control1 (Qubit): First control qubit.
        control2 (Qubit): Second control qubit.
        target (Qubit): Target qubit.
    """
    target.H()
    toffoli(control1, control2, target)
    target.H()


def n_qubit_controlled_u(
    controls_qubit: List[Qubit],
    context: ProgramContext,
    controlled_u_gate: Callable,
    target: Qubit,
) -> None:
    """Performs an n-qubit controlled-U gate with `controls_qubit` as controls
    and `target` as target, using ancillas qubit, Toffoli gates and a
    `controlled_u_gate` as controlled-U gate.

    The implementation is from "M. A. Nielsen and I. L. Chuang, Quantum Comput-
    ation and Quantum Information: 10th Anniversary Edition. Figure 4.10.".

    Args:
        controls_qubit (List[Qubit]): The list of n control qubits.
        context (ProgramContext): Context of the current program.
        controlled_u_gate (Callable): The controlled-U gate.
        target (Qubit): Target qubit.
    """

    n = len(controls_qubit)
    if n == 0 or n == 1:
        controlled_u_gate(controls_qubit[0], target)
        return

    ancillas = [Qubit(context.connection) for _ in range(n - 1)]

    toffoli(controls_qubit[0], controls_qubit[1], ancillas[0])
    for i in range(2, n):
        toffoli(controls_qubit[i], ancillas[i - 2], ancillas[i - 1])

    controlled_u_gate(ancillas[-1], target)

    for i in reversed(range(2, n)):
        toffoli(controls_qubit[i], ancillas[i - 2], ancillas[i - 1])
    toffoli(controls_qubit[0], controls_qubit[1], ancillas[0])

    for ancilla in ancillas:
        ancilla.measure()
