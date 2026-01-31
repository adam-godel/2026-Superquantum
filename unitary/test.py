import argparse
import os
import re
import numpy as np
import scipy.linalg

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.qasm3 import loads

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

RX = lambda theta: np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
RY = lambda theta: np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
RZ = lambda theta: np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])

expected = {
    1: np.block([
        [np.eye(2), np.zeros((2,2))],
        [np.zeros((2,2)), Y]
    ]),
    2: np.block([
        [np.eye(2), np.zeros((2,2))],
        [np.zeros((2,2)), RY(np.pi/7)]
    ]),
    3: scipy.linalg.expm(1j*np.pi/7*(np.kron(Z,Z))),
    4: scipy.linalg.expm(1j*np.pi/7*(np.kron(X,X)+np.kron(Y,Y))),
    5: scipy.linalg.expm(1j*np.pi/4*(np.kron(X,X)+np.kron(Y,Y)+np.kron(Z,Z))),
    6: scipy.linalg.expm(1j*np.pi/7*(np.kron(X,X)+np.kron(Z,np.eye(2))+np.kron(np.eye(2),Z)))
}

def load_qasm_circuit(path: str) -> tuple[QuantumCircuit, str]:
    with open(path, "r", encoding="utf-8") as f:
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
        dist = np.linalg.norm((phase_factor * actual) - expected)
        
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

def count_t_gates_manual(qasm_str: str) -> int:
    gate_t_counts = {}
    
    # Extract all gate definitions: gate NAME ... { BODY }
    gate_defs = re.findall(r"gate\s+(\w+).*?\{(.*?)\}", qasm_str, re.DOTALL)
    
    # We loop multiple times to resolve nesting (e.g., 42 calls 43)
    # For most generated circuits, 3-5 passes is plenty.
    for _ in range(5): 
        for name, body in gate_defs:
            # 1. Count direct 't' and 'tdg' in this definition
            current_count = len(re.findall(r"\b(t|tdg)\b", body))
            
            # 2. Add counts from OTHER custom gates called inside THIS body
            for other_name, other_count in gate_t_counts.items():
                if other_name != name:
                    # Find how many times 'other_name' is called in 'body'
                    calls = len(re.findall(rf"\b{other_name}\b", body))
                    current_count += (calls * other_count)
            
            gate_t_counts[name] = current_count

    # 3. Process the main execution block (everything outside 'gate' definitions)
    main_body = re.sub(r"gate.*?\{.*?\}", "", qasm_str, flags=re.DOTALL)
    
    # Start with direct T-calls in the main body
    total_t = len(re.findall(r"\b(t|tdg)\b", main_body))
    
    # Add counts from custom gate calls in the main body
    for name, count in gate_t_counts.items():
        calls = len(re.findall(rf"\b{name}\b", main_body))
        total_t += (calls * count)
        
    return total_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qasm_file", help="Path to QASM file (e.g., unitary1.qasm)")
    args = parser.parse_args()

    unitary_id = parse_unitary_id_from_filename(args.qasm_file)

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

    U_qasm = circuit_unitary(qc)
    
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

    aligned = distance_global_phase(U_qasm, U_expected)

    print("Best-aligned actual matrix (phase * actual, rounded to 6 decimals):")
    print(np.round(aligned, 6))
    print()

    err = np.linalg.norm(aligned - U_expected)
    print(f"Min |Δ|: {err:.3e}")

    # print t gate count
    t_count = count_t_gates_manual(qasm_src)
    print(f"T-gate count: {t_count}")

if __name__ == "__main__":
    main()
