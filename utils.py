from __future__ import annotations
from typing import List, Sequence, Union

from qiskit import QuantumCircuit

import mpmath
from pygridsynth.gridsynth import gridsynth_gates

GateSeq = Union[str, Sequence[str]]

def _tokenize(gates: GateSeq) -> List[str]:
    if isinstance(gates, str):
        s = gates.strip().replace(" ", "")
    else:
        s = "".join(str(g).strip() for g in gates)
    
    s = "".join(c for c in s.upper() if c in "HTSX")
    
    while "TTTTTTT" in s:
        s = s.replace("TTTTTTT", "t")
    
    return list(s)

def _apply_gate(qc: QuantumCircuit, g: str, i: int, dagger: bool) -> None:
    g_up = g.upper()
    if g_up == "H":
        qc.h(i)
        return

    def Rzhalf():
        qc.tdg(i) if dagger else qc.t(i)

    if g == "t":
        qc.tdg(i) if not dagger else qc.t(i)
        return
    
    if g_up == "T":
        Rzhalf()
        return

    if g_up == "S":
        Rzhalf(); Rzhalf()
        return

    if g_up == "X":
        qc.h(i)
        Rzhalf(); Rzhalf(); Rzhalf(); Rzhalf()
        qc.h(i)
        return


def gates_to_qiskit_circuit(gates: GateSeq, i: int, reverse: bool) -> QuantumCircuit:
    toks = _tokenize(gates)

    ordered = toks if reverse else list(reversed(toks))
    dagger = reverse

    qc = QuantumCircuit(i + 1)
    for g in ordered:
        _apply_gate(qc, g, i, dagger=dagger)
    return qc

def Rz(i: int, theta: float, epsilon: float) -> QuantumCircuit:
    mpmath.mp.dps = 128
    theta = mpmath.mpf(str(abs(theta)))
    epsilon = mpmath.mpf(str(epsilon))

    gates = gridsynth_gates(theta=theta, epsilon=epsilon)

    reverse = theta < 0
    return gates_to_qiskit_circuit(gates, i, reverse)