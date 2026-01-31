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
        [np.exp(-1j*np.pi/7), 0, 0, 0],
        [0, (np.exp(3*1j*np.pi/7)+np.exp(-1j*np.pi/7))/2, (np.exp(3*1j*np.pi/7)-np.exp(-1j*np.pi/7))/2, 0],
        [0, (np.exp(3*1j*np.pi/7)-np.exp(-1j*np.pi/7))/2, (np.exp(3*1j*np.pi/7)+np.exp(-1j*np.pi/7))/2, 0],
        [0, 0, 0, np.exp(-1j*np.pi/7)]
    ])
}

def load_qasm_circuit(path: str) -> QuantumCircuit:
    with open(path, "r") as f:
        qasm3_src = f.read()

    qc = loads(qasm3_src)
    return qc, qasm3_src

def circuit_unitary(qc: QuantumCircuit) -> np.ndarray:
    return Operator(qc).data

def distance_global_phase(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    best_phase = 1.0 + 0.0j
    min_dist = float('inf')

    for phase in np.arange(-2 * np.pi, 2 * np.pi, .001):
        phase_factor = np.exp(1j * phase)
        dist = np.linalg.norm((phase_factor * actual) -  expected)
        
        if dist < min_dist:
            min_dist = dist
            best_phase = phase_factor

    print(f"Best phase found: {best_phase:.4f} (Distance: {min_dist:.2e})")
    aligned_matrix = best_phase * actual
    
    return aligned_matrix

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

    qc, _ = load_qasm_circuit(args.qasm_file)

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

    aligned = distance_global_phase(U_expected, U_qasm, atol=args.atol)

    print("Best-aligned actual matrix (phase * actual, rounded to 6 decimals):")
    print(np.round(aligned, 6))
    print()

    err = np.max(np.abs(U_expected - aligned))
    print(f"Max |Δ|: {err:.3e}")

    # print t gate count
    t_count = count_t_gates(qc)
    print(f"T-gate count: {t_count}")

if __name__ == "__main__":
    main()
