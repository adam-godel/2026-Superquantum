import argparse
import os
import re
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.qasm3 import loads

def load_qasm_circuit(path: str) -> QuantumCircuit:
    """
    Load an OpenQASM 3 file into a Qiskit QuantumCircuit.
    """

    with open(path, "r", encoding="utf-8") as f:
        qasm3_src = f.read()

    qc = loads(qasm3_src)
    return qc


def circuit_unitary(qc: QuantumCircuit) -> np.ndarray:
    # Operator(qc) constructs the full unitary for circuits with no measurement/reset.
    return Operator(qc).data


def equal_up_to_global_phase(U: np.ndarray, V: np.ndarray, atol=1e-8) -> tuple[bool, complex]:
    """
    Returns (is_equal, phase_factor) where phase_factor is a complex scalar e^{iθ}
    such that U ≈ phase_factor * V if is_equal is True.
    """
    if U.shape != V.shape:
        return False, 1.0 + 0.0j

    # Find a reference element where V is nonzero to estimate phase robustly
    idx = None
    flatV = V.ravel()
    for k in range(flatV.size):
        if abs(flatV[k]) > atol:
            idx = k
            break

    if idx is None:
        # V is basically all zeros (shouldn't happen for a unitary)
        return np.allclose(U, V, atol=atol), 1.0 + 0.0j

    phase = U.ravel()[idx] / V.ravel()[idx]
    if abs(phase) > 0:
        phase = phase / abs(phase)  # normalize to unit magnitude

    return np.allclose(U, phase * V, atol=atol), phase


def parse_unitary_id_from_filename(path: str) -> int:
    """
    Extracts an integer id from filenames like:
    unitary1.qasm, unitary2.qasm, my_unitary_12.qasm, etc.
    If you prefer explicit ids, pass --id instead.
    """
    base = os.path.basename(path)
    m = re.search(r"(\d+)", base)
    if not m:
        raise ValueError(f"Could not infer unitary id from filename: {base}. Use --id.")
    return int(m.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qasm_file", help="Path to QASM file (e.g., unitary1.qasm)")
    parser.add_argument("--id", type=int, default=None, help="Unitary id override")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for comparisons")
    args = parser.parse_args()

    unitary_id = args.id if args.id is not None else parse_unitary_id_from_filename(args.qasm_file)

    # ---- 1) Define / load your expected matrices here ----
    # Example placeholder dictionary. Replace with your real matrices.
    # Each entry must be a (2**n x 2**n) complex numpy array.
    expected = {
        1: np.eye(4, dtype=complex),
    }


    if unitary_id not in expected:
        raise KeyError(
            f"Unitary id {unitary_id} not found in expected dict. "
            f"Available keys: {sorted(expected.keys())}"
        )

    U_expected = np.asarray(expected[unitary_id], dtype=complex)

    # ---- 2) Load QASM -> circuit ----
    qc = load_qasm_circuit(args.qasm_file)

    # Guardrail: Operator needs unitary-only circuit (no measurements/resets)
    if qc.num_clbits > 0:
        # Measurements can exist even if clbits are unused; check instructions.
        inst_names = [inst.operation.name for inst in qc.data]
        if "measure" in inst_names or "reset" in inst_names:
            raise ValueError("Circuit contains measure/reset; cannot form a single unitary Operator.")

    # ---- 3) Circuit -> unitary ----
    U_qasm = circuit_unitary(qc)

    # ---- 4) Sanity: dimensions match ----
    if U_qasm.shape != U_expected.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"  from QASM:     {U_qasm.shape}\n"
            f"  expected dict: {U_expected.shape}\n"
            f"QASM qubits: {qc.num_qubits} -> expected dimension {2**qc.num_qubits}"
        )

    # ---- 5) Compare ----
    direct_ok = np.allclose(U_qasm, U_expected, atol=args.atol)
    phase_ok, phase = equal_up_to_global_phase(U_qasm, U_expected, atol=args.atol)

    print(f"QASM file: {args.qasm_file}")
    print(f"Inferred id: {unitary_id}")
    print(f"Qubits: {qc.num_qubits}")
    print(f"Matrix shape: {U_qasm.shape}")
    print()
    print(f"allclose (direct): {direct_ok}")
    print(f"allclose (up to global phase): {phase_ok}")
    if phase_ok and not direct_ok:
        print(f"Estimated global phase factor (U_qasm ≈ phase * U_expected): {phase}")

    # Optional: show max entrywise error under best phase alignment
    if phase_ok:
        err = np.max(np.abs(U_qasm - phase * U_expected))
        print(f"Max |Δ| after phase alignment: {err:.3e}")
    else:
        err = np.max(np.abs(U_qasm - U_expected))
        print(f"Max |Δ| (no phase alignment): {err:.3e}")


if __name__ == "__main__":
    main()
