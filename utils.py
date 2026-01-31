"""
pygridsynth-style gate string/list -> Qiskit commands with custom decompositions.

Default behavior:
- Applies gates RIGHT -> LEFT (as requested).
- Uses the custom mappings:
    H -> qc.h(i)
    T -> qc.t(i)
    S -> qc.t(i); qc.t(i)                # S = T^2
    X -> qc.h(i); qc.t(i)*4; qc.h(i)     # X = H Z H, and Z = T^4

If reverse=True:
- Applies gates LEFT -> RIGHT
- Uses dagger versions of the mapped gates:
    H -> qc.h(i)                         # self-adjoint
    T -> qc.tdg(i)
    S -> qc.tdg(i); qc.tdg(i)            # (T^2)† = (T†)^2
    X -> qc.h(i); qc.tdg(i)*4; qc.h(i)   # X† = X

Input accepted:
- list[str] like ["H","T","S","X", ...]
- string like "H T S X" or "HTSX" (compact mode splits into single chars)

This script returns a QuantumCircuit and also can emit the literal command lines if desired.
"""

from __future__ import annotations
from typing import Iterable, List, Sequence, Union
import re

from qiskit import QuantumCircuit


GateSeq = Union[str, Sequence[str]]


def _tokenize(gates: GateSeq) -> List[str]:
    if isinstance(gates, str):
        s = gates.strip().replace(" ", "")  # Remove all spaces
        # Filter to only keep HTSX characters
        s = "".join(c for c in s.upper() if c in "HTSX")
        return list(s)
    # For sequences, filter and clean each element
    return [str(g).strip().upper() for g in gates if str(g).strip().upper() in "HTSX"]


def _apply_gate(qc: QuantumCircuit, g: str, i: int, *, dagger: bool) -> None:
    g_up = g.upper()

    if g_up == "H":
        qc.h(i)
        return

    # pick T vs Tdg primitive based on dagger flag
    def T1():
        qc.tdg(i) if dagger else qc.t(i)

    if g_up == "T":
        T1()
        return

    if g_up == "S":
        # S = T^2 ; S† = (T†)^2
        T1(); T1()
        return

    if g_up == "X":
        # X = H Z H, and Z = T^4 ; X† = X
        qc.h(i)
        T1(); T1(); T1(); T1()
        qc.h(i)
        return


def gates_to_qiskit_circuit(gates: GateSeq, i: int, *, reverse: bool = False) -> QuantumCircuit:
    """
    Parameters
    ----------
    gates : str | Sequence[str]
        Gate sequence. Example: "H T S X" or "HTSX" or ["H","T","S","X"].
    i : int
        Qubit index to apply all gates to.
    reverse : bool
        If False (default): apply RIGHT->LEFT using non-dagger mappings.
        If True: apply LEFT->RIGHT using dagger mappings (H same; T->tdg; etc.).

    Returns
    -------
    QuantumCircuit
        1-qubit circuit with operations applied to qubit i (requires qc has at least i+1 qubits).
    """
    toks = _tokenize(gates)

    # default is RIGHT -> LEFT
    ordered = toks if reverse else list(reversed(toks))
    dagger = reverse

    qc = QuantumCircuit(i + 1)
    for g in ordered:
        _apply_gate(qc, g, i, dagger=dagger)
    return qc


def gates_to_qiskit_lines(gates: GateSeq, i: int, *, reverse: bool = False) -> List[str]:
    """
    Same logic as `gates_to_qiskit_circuit`, but returns the literal command lines
    like 'qc.h(i)', 'qc.t(i)', 'qc.tdg(i)', etc.
    """
    toks = _tokenize(gates)
    ordered = toks if reverse else list(reversed(toks))
    dagger = reverse

    lines: List[str] = []

    def Tline():
        lines.append(f"qc.tdg({i})" if dagger else f"qc.t({i})")

    for g in ordered:
        gu = g.upper()
        if gu == "H":
            lines.append(f"qc.h({i})")
        elif gu == "T":
            Tline()
        elif gu == "S":
            Tline(); Tline()
        elif gu == "X":
            lines.append(f"qc.h({i})")
            Tline(); Tline(); Tline(); Tline()
            lines.append(f"qc.h({i})")

    return lines


# --- quick demo ---
if __name__ == "__main__":
    seq = "H T S X"

    print("Default (RIGHT->LEFT, non-dagger):")
    for line in gates_to_qiskit_lines(seq, i=0, reverse=False):
        print(line)

    print("\nReverse=True (LEFT->RIGHT, dagger):")
    for line in gates_to_qiskit_lines(seq, i=0, reverse=True):
        print(line)

    qc = gates_to_qiskit_circuit(seq, i=0, reverse=False)
    print("\nCircuit:")
    print(qc.draw())
