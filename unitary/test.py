import argparse
import os
import re
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.qasm3 import loads

# ---- 1) Define / load your expected matrices here ----
# Example placeholder dictionary. Replace with your real matrices.
# Each entry must be a (2**n x 2**n) complex numpy array.
expected = {
    1: np.array([
        [1, 0, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1, 0],
        [0, 1j, 0, 0]
    ]),
    2: np.array([
        [1, 0, 0, 0],
        [0, np.cos(np.pi / 14), 0, -np.sin(np.pi / 14)],
        [0, 0, 1, 0],
        [0, np.sin(np.pi / 14), 0, np.cos(np.pi / 14)]
    ]),
    3: np.array([
        [np.exp(-1j*np.pi/7), 0, 0, 0],
        [0, np.exp(1j*np.pi/7), 0, 0],
        [0, 0, np.exp(1j*np.pi/7), 0],
        [0, 0, 0, np.exp(-1j*np.pi/7)]
    ]),
    4: np.array([
        [1, 0, 0, 0],
        [0, np.cos(2*np.pi/7), 1j*np.sin(2*np.pi/7), 0],
        [0, 1j*np.sin(2*np.pi/7), np.cos(2*np.pi/7), 0],
        [0, 0, 0, 1]
    ]),
    5: np.array([
        
    ])
}

def load_qasm_circuit(path: str) -> tuple[QuantumCircuit, str]:
    """
    Load an OpenQASM 3 file into a Qiskit QuantumCircuit.
    Returns both the circuit and the original QASM source.
    """

    with open(path, "r", encoding="utf-8") as f:
        qasm3_src = f.read()

    qc = loads(qasm3_src)
    return qc, qasm3_src


def circuit_unitary(qc: QuantumCircuit) -> np.ndarray:
    # Operator(qc) constructs the full unitary for circuits with no measurement/reset.
    return Operator(qc).data

def equal_up_to_global_phase(U: np.ndarray, V: np.ndarray, atol=1e-8) -> tuple[bool, complex, np.ndarray]:
    """
    Returns (is_equal, phase_factor, aligned_matrix) where:
    - phase_factor is a complex scalar e^{iθ} such that U ≈ phase_factor * V
    - aligned_matrix is phase_factor * V (the best-aligned version of V)
    """
    if U.shape != V.shape:
        return False, 1.0 + 0.0j, V

    mask = (np.abs(V) > atol) & (np.abs(U) > atol)

    if not np.any(mask):
        if np.allclose(U, 0, atol=atol) and np.allclose(V, 0, atol=atol):
            return True, 1.0 + 0.0j, V
        return False, 1.0 + 0.0j, V

    raw_ratios = U[mask] / V[mask]

    candidates = raw_ratios / np.abs(raw_ratios)

    best_phase = 1.0 + 0.0j
    min_dist = float('inf')

    for phase in candidates:
        dist = np.linalg.norm(U - (phase * V))
        
        if dist < min_dist:
            min_dist = dist
            best_phase = phase

    print(f"Best phase found: {best_phase:.4f} (Distance: {min_dist:.2e})")

    # Calculate the best-aligned matrix
    aligned_matrix = best_phase * V
    
    # Final strict check using the best phase found
    is_equal = np.allclose(U, aligned_matrix, atol=atol)
    return is_equal, best_phase, aligned_matrix


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

def count_t_gates(qc: QuantumCircuit) -> int:
    """
    Decomposes custom gates and counts all 't' and 'tdg' gates accurately.
    """
    # Use decompose to unroll custom 'gate' definitions into standard gates
    decomposed_qc = qc.decompose()
    
    # In case of nested definitions, you might need multiple passes or 
    # use the transpiler to unroll to a specific basis
    ops = decomposed_qc.count_ops()
    
    t_count = ops.get("t", 0) + ops.get("tdg", 0)
    return t_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qasm_file", help="Path to QASM file (e.g., unitary1.qasm)")
    parser.add_argument("--id", type=int, default=None, help="Unitary id override")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for comparisons")
    args = parser.parse_args()

    unitary_id = args.id if args.id is not None else parse_unitary_id_from_filename(args.qasm_file)

    if unitary_id not in expected:
        raise KeyError(
            f"Unitary id {unitary_id} not found in expected dict. "
            f"Available keys: {sorted(expected.keys())}"
        )

    U_expected = np.asarray(expected[unitary_id], dtype=complex)

    # ---- 2) Load QASM -> circuit ----
    qc, qasm_src = load_qasm_circuit(args.qasm_file)

    # Guardrail: Operator needs unitary-only circuit (no measurements/resets)
    if qc.num_clbits > 0:
        # Measurements can exist even if clbits are unused; check instructions.
        inst_names = [inst.operation.name for inst in qc.data]
        if "measure" in inst_names or "reset" in inst_names:
            raise ValueError("Circuit contains measure/reset; cannot form a single unitary Operator.")

    # ---- 3) Circuit -> unitary ----
    U_qasm = circuit_unitary(qc).T
    
    print("Expected matrix:")
    print(np.round(U_expected, 6))
    print()
    print("Matrix from QASM file (rounded to 6 decimals):")
    print(np.round(U_qasm, 6))
    print()

    # ---- 4) Sanity: dimensions match ----
    if U_qasm.shape != U_expected.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"  from QASM:     {U_qasm.shape}\n"
            f"  expected dict: {U_expected.shape}\n"
            f"QASM qubits: {qc.num_qubits} -> expected dimension {2**qc.num_qubits}"
        )

    # ---- 5) Compare matrices ----
    # First try direct comparison
    direct_ok = np.allclose(U_qasm, U_expected, atol=args.atol)
    
    # Then try comparison up to global phase
    phase_ok, phase, aligned_expected = equal_up_to_global_phase(U_qasm, U_expected, atol=args.atol)

    # Always print the best-aligned expected matrix for comparison
    print("Best-aligned expected matrix (phase * expected, rounded to 6 decimals):")
    print(np.round(aligned_expected, 6))
    print()

    # Print results
    print(f"QASM file: {args.qasm_file}")
    print(f"Inferred id: {unitary_id}")
    print(f"Qubits: {qc.num_qubits}")
    print(f"Matrix shape: {U_qasm.shape}")
    print()
    print(f"allclose (direct): {direct_ok}")
    print(f"allclose (up to global phase): {phase_ok}")
    if phase_ok and not direct_ok:
        print(f"Estimated global phase factor: {phase}")
    elif not direct_ok:
        print(f"Best phase factor found: {phase} (but matrices still don't match within tolerance)")

    # Show max error
    if phase_ok:
        err = np.max(np.abs(U_qasm - aligned_expected))
        print(f"Max |Δ| after phase alignment: {err:.3e}")
    else:
        err = np.max(np.abs(U_qasm - U_expected))
        print(f"Max |Δ| (no phase alignment): {err:.3e}")

    # print t gate count
    t_count = count_t_gates(qc)
    print(f"T-gate count: {t_count}")

    # Final result
    if phase_ok:
        print(f"\n✓ SUCCESS: Matrices match up to global phase")
    elif direct_ok:
        print(f"\n✓ SUCCESS: Matrices match exactly")
    else:
        print(f"\n✗ FAILURE: Matrices do not match")

if __name__ == "__main__":
    main()
