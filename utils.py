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

def _apply_gate(qc: QuantumCircuit, g: str, i: int, *, dagger: bool) -> None:
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


def gates_to_qiskit_circuit(gates: GateSeq, i: int, *, reverse: bool = False) -> QuantumCircuit:
    toks = _tokenize(gates)

    ordered = toks if reverse else list(reversed(toks))
    dagger = reverse

    qc = QuantumCircuit(i + 1)
    for g in ordered:
        _apply_gate(qc, g, i, dagger=dagger)
    return qc


def gates_to_qiskit_lines(gates: GateSeq, i: int, *, reverse: bool = False) -> List[str]:
    toks = _tokenize(gates)
    ordered = toks if reverse else list(reversed(toks))
    dagger = reverse

    lines: List[str] = []

    def Rzhalf():
        lines.append(f"qc.tdg({i})" if dagger else f"qc.t({i})")

    for g in ordered:
        gu = g.upper()
        if gu == "H":
            lines.append(f"qc.h({i})")
        elif g == "t":
            lines.append(f"qc.t({i})" if dagger else f"qc.tdg({i})")
        elif gu == "T":
            Rzhalf()
        elif gu == "S":
            Rzhalf(); Rzhalf()
        elif gu == "X":
            lines.append(f"qc.h({i})")
            Rzhalf(); Rzhalf(); Rzhalf(); Rzhalf()
            lines.append(f"qc.h({i})")

    return lines

def Rz(theta: float, epsilon: float):
    mpmath.mp.dps = 128
    theta = mpmath.mpf(str(abs(theta)))
    epsilon = mpmath.mpf(str(epsilon))  
    return gridsynth_gates(theta=theta, epsilon=epsilon)