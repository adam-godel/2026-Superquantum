from __future__ import annotations
from typing import List

from qiskit import QuantumCircuit

import mpmath
from pygridsynth.gridsynth import gridsynth_gates

def _tokenize(gates: str) -> List[str]:
    s = gates.strip().replace(" ", "")
    s = "".join(c for c in s.upper() if c in "HTSX")
    
    while "TTTTTTT" in s:
        s = s.replace("TTTTTTT", "t")
    
    return list(s)

def _apply_gate(qc: QuantumCircuit, g: str, dagger: bool) -> None:
    g_up = g.upper()
    if g_up == "H":
        qc.h(0)
        return

    if g == "t":
        qc.tdg(0) if not dagger else qc.t(0)
        return
    
    if g_up == "T":
        qc.tdg(0) if dagger else qc.t(0)
        return

    if g_up == "S":
        qc.sdg(0) if dagger else qc.s(0)
        return

    if g_up == "X":
        qc.h(0)
        qc.s(0); qc.s(0)
        qc.h(0)
        return


def gates_to_qiskit_circuit(gates: str, reverse: bool) -> QuantumCircuit:
    toks = _tokenize(gates)

    ordered = toks if reverse else list(reversed(toks))
    dagger = reverse

    qc = QuantumCircuit(1)
    for g in ordered:
        _apply_gate(qc, g, dagger)
    return qc

def Rz(theta: float, epsilon: float) -> QuantumCircuit:
    reverse = theta < 0
    
    mpmath.mp.dps = 128
    theta = mpmath.mpf(str(abs(theta)))
    epsilon = mpmath.mpf(str(epsilon))

    gates = gridsynth_gates(theta=theta, epsilon=epsilon)

    return gates_to_qiskit_circuit(gates, reverse)

def Ry(theta: float, epsilon: float) -> QuantumCircuit:
    qc = QuantumCircuit(1)

    qc.sdg(0)
    qc.h(0)
    qc.append(Rz(theta, epsilon).to_gate(), [0])
    qc.h(0)
    qc.s(0)

    return qc

def Rx(theta: float, epsilon: float) -> QuantumCircuit:
    qc = QuantumCircuit(1)

    qc.h(0)
    qc.append(Rz(theta, epsilon).to_gate(), [0])
    qc.h(0)

    return qc