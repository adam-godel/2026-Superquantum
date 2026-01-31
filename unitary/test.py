import argparse
import os
import re
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.qasm3 import loads

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
    with open(path, "r", encoding="utf-8") as f:
        qasm3_src = f.read()

    qc = loads(qasm3_src)
    return qc, qasm3_src


def circuit_unitary(qc: QuantumCircuit) -> np.ndarray:
    return Operator(qc).data

def equal_up_to_global_phase(U: np.ndarray, V: np.ndarray, atol=1e-8) -> tuple[bool, complex, np.ndarray]:
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
    aligned_matrix = best_phase * V
    
    is_equal = np.allclose(U, aligned_matrix, atol=atol)
    return is_equal, best_phase, aligned_matrix


def parse_unitary_id_from_filename(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"(\d+)", base)
    if not m:
        raise ValueError(f"Could not infer unitary id from filename: {base}. Use --id.")
    return int(m.group(1))

def count_t_gates(qc: QuantumCircuit) -> int:
    decomposed_qc = qc.decompose()
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

    qc, qasm_src = load_qasm_circuit(args.qasm_file)

    if qc.num_clbits > 0:
        inst_names = [inst.operation.name for inst in qc.data]
        if "measure" in inst_names or "reset" in inst_names:
            raise ValueError("Circuit contains measure/reset; cannot form a single unitary Operator.")

    U_qasm = circuit_unitary(qc).T
    
    print("Expected matrix:")
    print(np.round(U_expected, 6))
    print()

    if U_qasm.shape != U_expected.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"  from QASM:     {U_qasm.shape}\n"
            f"  expected dict: {U_expected.shape}\n"
            f"QASM qubits: {qc.num_qubits} -> expected dimension {2**qc.num_qubits}"
        )

    direct_ok = np.allclose(U_qasm, U_expected, atol=args.atol)
    
    phase_ok, phase, aligned_expected = equal_up_to_global_phase(U_qasm, U_expected, atol=args.atol)

    print("Best-aligned expected matrix (phase * expected, rounded to 6 decimals):")
    print(np.round(aligned_expected, 6))
    print()

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

if __name__ == "__main__":
    main()
