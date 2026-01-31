#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


# ----------------------------
# Parsing + circuit building
# ----------------------------

@dataclass(frozen=True)
class ParsedGate:
    name: str   # normalized gate name: H, T, S, X, TDG, SDG
    qubit: int


def _strip_comments(line: str) -> str:
    # Remove inline // or # comments too
    line = re.split(r"(//|#)", line, maxsplit=1)[0]
    # Some formats use ';' as comment starter
    line = re.split(r";", line, maxsplit=1)[0]
    return line.strip()


def _normalize_gate_name(raw: str) -> str:
    s = raw.strip()

    # normalize unicode dagger
    s = s.replace("†", "DG")
    s_up = s.upper()

    # common aliases
    aliases = {
        "TADG": "TDG",
        "TDAG": "TDG",
        "T_DG": "TDG",
        "SDAG": "SDG",
        "SADG": "SDG",
        "S_DG": "SDG",
    }
    s_up = aliases.get(s_up, s_up)

    # Allow "H", "T", "S", "X", "TDG", "SDG"
    return s_up


def read_qasm(path: str) -> List[ParsedGate]:
    gates: List[ParsedGate] = []

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = _strip_comments(raw_line)
            if not line:
                continue

            # Extract gate token: leading letters (plus optional dagger symbol already normalized later)
            m_gate = re.match(r"\s*([A-Za-z]+|[A-Za-z]†)\b", line)
            if not m_gate:
                raise ValueError(f"{path}:{lineno}: Could not parse gate token from line: {raw_line.rstrip()}")

            gate_raw = m_gate.group(1)
            gate = _normalize_gate_name(gate_raw)

            # Extract first integer on the line as qubit index (supports q[0], q0, etc.)
            m_q = re.search(r"(-?\d+)", line)
            if not m_q:
                raise ValueError(f"{path}:{lineno}: Could not find qubit index integer in line: {raw_line.rstrip()}")

            q = int(m_q.group(1))

            if gate not in {"H", "T", "S", "X", "TDG", "SDG"}:
                raise ValueError(
                    f"{path}:{lineno}: Unsupported gate '{gate_raw}' (normalized: {gate}). "
                    f"Supported: H, T, S, X, TDG, SDG."
                )

            if q < 0:
                raise ValueError(f"{path}:{lineno}: Qubit index must be >= 0, got {q}.")

            gates.append(ParsedGate(gate, q))

    return gates


def build_circuit_from_parsed(gates: List[ParsedGate], reverse: bool = False) -> QuantumCircuit:
    if not gates:
        raise ValueError("No gates found in QASM file.")

    max_q = max(g.qubit for g in gates)
    qc = QuantumCircuit(max_q + 1)

    ordered = gates if reverse else list(reversed(gates))

    for g in ordered:
        name = g.name
        i = g.qubit

        def T1(dagger: bool) -> None:
            qc.tdg(i) if dagger else qc.t(i)

        if name == "H":
            qc.h(i)

        elif name == "T":
            qc.t(i)

        elif name == "TDG":
            qc.tdg(i)

        elif name == "S":
            # S = T^2
            qc.t(i); qc.t(i)

        elif name == "SDG":
            # S† = (T†)^2
            qc.tdg(i); qc.tdg(i)

        elif name == "X":
            # X = H Z H, Z = T^4
            qc.h(i)
            qc.t(i); qc.t(i); qc.t(i); qc.t(i)
            qc.h(i)

        else:
            raise ValueError(f"Internal error: unhandled gate {name!r}")

    return qc


# ----------------------------
# Unitary / “closed” tests
# ----------------------------

def is_close(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> bool:
    return np.allclose(a, b, atol=atol, rtol=rtol)


def test_unitary(U: np.ndarray, atol: float, rtol: float) -> Tuple[bool, str]:
    n = U.shape[0]
    I = np.eye(n, dtype=complex)

    U_dag = U.conj().T
    left = U_dag @ U
    right = U @ U_dag

    ok_left = is_close(left, I, atol=atol, rtol=rtol)
    ok_right = is_close(right, I, atol=atol, rtol=rtol)

    if not (ok_left and ok_right):
        # helpful diagnostic: worst deviation
        dev_left = np.max(np.abs(left - I))
        dev_right = np.max(np.abs(right - I))
        return False, f"Not unitary within tolerances. max|U†U-I|={dev_left:.3e}, max|UU†-I|={dev_right:.3e}"

    # “closed” under adjoint: inverse equals adjoint for unitary matrices
    try:
        U_inv = np.linalg.inv(U)
    except np.linalg.LinAlgError:
        return False, "Matrix inversion failed (singular)."

    ok_inv = is_close(U_inv, U_dag, atol=atol, rtol=rtol)
    if not ok_inv:
        dev_inv = np.max(np.abs(U_inv - U_dag))
        return False, f"Inverse not equal to adjoint within tolerances. max|inv(U)-U†|={dev_inv:.3e}"

    # determinant magnitude check (optional sanity)
    det = np.linalg.det(U)
    if not np.isfinite(det.real) or not np.isfinite(det.imag):
        return False, "Determinant is not finite."
    if not np.isclose(abs(det), 1.0, atol=atol*10, rtol=rtol*10):
        return False, f"|det(U)| not ~ 1. Got |det(U)|={abs(det):.6f}"

    return True, "PASS: Matrix is unitary and closed under adjoint (inv(U) ≈ U†)."


def main() -> None:
    parser = argparse.ArgumentParser(description="QASM -> Qiskit -> matrix unitary/closure test")
    parser.add_argument("qasm_file", help="Path to unitary*.qasm file")
    parser.add_argument("--reverse", action="store_true",
                        help="If set: apply LEFT->RIGHT (no implicit reversal). Default applies RIGHT->LEFT.")
    parser.add_argument("--atol", type=float, default=1e-10, help="Absolute tolerance for numeric checks")
    parser.add_argument("--rtol", type=float, default=1e-10, help="Relative tolerance for numeric checks")
    parser.add_argument("--print-circuit", action="store_true", help="Print the circuit diagram")
    parser.add_argument("--print-matrix", action="store_true", help="Print the unitary matrix (can be large)")
    args = parser.parse_args()

    parsed = read_qasm(args.qasm_file)
    qc = build_circuit_from_parsed(parsed, reverse=args.reverse)

    if args.print_circuit:
        print(qc.draw())

    U = np.array(Operator(qc).data, dtype=complex)

    if args.print_matrix:
        np.set_printoptions(linewidth=200, precision=6, suppress=True)
        print(U)

    ok, msg = test_unitary(U, atol=args.atol, rtol=args.rtol)

    # Some extra context
    print(f"File: {args.qasm_file}")
    print(f"Qubits: {qc.num_qubits}")
    print(f"Gates parsed: {len(parsed)}")
    print(f"Order: {'LEFT->RIGHT' if args.reverse else 'RIGHT->LEFT'}")
    print(msg)

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()