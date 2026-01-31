"""
pygridsynth -> qiskit converter

Supports several common "gate list" encodings:
- list[str] like ["H","T","S","H",...]
- list[tuple] like [("H",0),("T",0),...]
- a single string like "H T S H" or "HTSH" (best-effort)
- nested lists (will be flattened)

Gate-name normalization supported:
H, X, Y, Z, S, SDG/SDAG/S†, T, TDG/TDAG/T†, I/ID
Also supports CX/CNOT if tuples specify (name, control, target).

If your pygridsynth object is some custom class, pass `str(gates)` and it will try to parse.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence, Tuple, Union, Optional

from qiskit import QuantumCircuit


GateToken = Union[str, Tuple[Any, ...]]


# --- parsing helpers ---------------------------------------------------------

def _flatten(x: Any) -> List[Any]:
    if isinstance(x, (list, tuple)):
        out: List[Any] = []
        for y in x:
            out.extend(_flatten(y))
        return out
    return [x]


def _normalize_name(name: str) -> str:
    n = name.strip()

    # common unicode dagger variants
    n = n.replace("†", "DG").replace("⁻¹", "DG")

    n_up = n.upper()

    # map aliases
    alias = {
        "CNOT": "CX",
        "CN": "CX",
        "CONTROLLEDNOT": "CX",
        "S_DG": "SDG",
        "SADG": "SDG",
        "SDAG": "SDG",
        "T_DG": "TDG",
        "TADG": "TDG",
        "TDAG": "TDG",
        "IDEN": "I",
        "IDENTITY": "I",
    }
    n_up = alias.get(n_up, n_up)

    # sometimes people write like "Sdg" / "Tdg"
    n_up = n_up.replace("SDG", "SDG").replace("TDG", "TDG")
    return n_up


def _tokenize_from_string(s: str) -> List[str]:
    """
    Best-effort:
    - If string contains separators/spaces/commas/brackets, extract words.
    - Else, if it's a compact string like "HTSH", split into single-letter gates.
      (Does NOT attempt to detect multi-letter tokens in compact mode.)
    """
    ss = s.strip()

    # If looks like Python list/tuple or contains separators, extract tokens
    if re.search(r"[\s,\[\]\(\);]", ss):
        # capture things like H, T, Sdg, T†, CX, CNOT
        raw = re.findall(r"[A-Za-z]+(?:\^?\+?DG)?|[STHXYZI]|[ST]†|[ST]dg|CNOT|CX", ss)
        if raw:
            return [t for t in raw if t.strip()]
        # fallback: split on whitespace/commas
        return [t for t in re.split(r"[\s,]+", ss) if t]

    # Compact mode (e.g., "HTSH")
    return list(ss)


def _coerce_gate_tokens(gates: Any) -> List[GateToken]:
    """
    Convert `gates` into a flat list of either:
    - strings (gate names)
    - tuples (gate name + qubit indices)
    """
    if gates is None:
        return []

    # If it's already a list/tuple-like of tokens
    if isinstance(gates, (list, tuple)):
        flat = _flatten(gates)
        return flat  # keep as-is; we’ll interpret each element below

    # If it’s a custom object, try iterating
    try:
        iter(gates)  # type: ignore
        if not isinstance(gates, (str, bytes)):
            return _flatten(list(gates))  # type: ignore
    except TypeError:
        pass

    # Fallback: parse from string representation
    return _tokenize_from_string(str(gates))


# --- qiskit builder ----------------------------------------------------------

def pygridsynth_to_qiskit(
    gates: Any,
    num_qubits: int = 1,
    target: int = 0,
    *,
    reverse: bool = False,
    add_barriers: bool = False,
    name: str = "pygridsynth",
) -> QuantumCircuit:
    """
    Build a Qiskit QuantumCircuit from a pygridsynth-style gate sequence.

    Parameters
    ----------
    gates:
        Output of `pygridsynth.gridsynth.gridsynth_gates(...)`, OR a list/string as described above.
    num_qubits:
        Total number of qubits in the output circuit.
    target:
        Default target qubit index for 1-qubit gates if not specified in tokens.
    reverse:
        If True, apply gates in reverse order (useful if your gate list is “right-to-left”).
    add_barriers:
        If True, add barriers between gates (debug visualization).
    name:
        Circuit name.

    Returns
    -------
    QuantumCircuit
    """
    qc = QuantumCircuit(num_qubits, name=name)

    tokens = _coerce_gate_tokens(gates)
    if reverse:
        tokens = list(reversed(tokens))

    def apply_1q(gname: str, q: int) -> None:
        g = _normalize_name(gname)
        if g in ("I", "ID"):
            return
        if g == "H":
            qc.h(q)
        elif g == "X":
            qc.x(q)
        elif g == "Y":
            qc.y(q)
        elif g == "Z":
            qc.z(q)
        elif g == "S":
            qc.s(q)
        elif g == "SDG":
            qc.sdg(q)
        elif g == "T":
            qc.t(q)
        elif g == "TDG":
            qc.tdg(q)
        else:
            raise ValueError(f"Unsupported/unknown 1-qubit gate token: {gname!r} (normalized: {g})")

    def apply_2q(gname: str, c: int, t: int) -> None:
        g = _normalize_name(gname)
        if g == "CX":
            qc.cx(c, t)
        else:
            raise ValueError(f"Unsupported/unknown 2-qubit gate token: {gname!r} (normalized: {g})")

    for tok in tokens:
        if add_barriers:
            qc.barrier()

        # tuple-like: ("H", 0) or ("CX", 0, 1)
        if isinstance(tok, tuple):
            if len(tok) == 0:
                continue
            gname = str(tok[0])

            if len(tok) == 2:
                apply_1q(gname, int(tok[1]))
            elif len(tok) == 3:
                apply_2q(gname, int(tok[1]), int(tok[2]))
            else:
                raise ValueError(f"Unsupported tuple token arity {len(tok)}: {tok!r}")
            continue

        # string token
        if isinstance(tok, (str, bytes)):
            s = tok.decode() if isinstance(tok, bytes) else tok
            s = s.strip()
            if not s:
                continue

            # if user passed one big string token (like "H T S"), split it
            if len(s) > 1 and re.search(r"[\s,\[\]\(\);]", s):
                subtoks = _tokenize_from_string(s)
                # re-run each subtoken as a simple 1q gate by default
                for st in subtoks:
                    apply_1q(st, target)
                continue

            apply_1q(s, target)
            continue

        # unknown token type
        raise TypeError(f"Unrecognized token type: {type(tok)} -> {tok!r}")

    return qc


# --- example usage -----------------------------------------------------------
if __name__ == "__main__":
    import mpmath
    from pygridsynth.gridsynth import gridsynth_gates

    mpmath.mp.dps = 128
    theta = mpmath.mpf("0.5")
    epsilon = mpmath.mpf("1e-10")

    gates = gridsynth_gates(theta=theta, epsilon=epsilon)
    print("pygridsynth gates:", gates)

    qc = pygridsynth_to_qiskit(gates, num_qubits=1, target=0)
    print(qc.draw())