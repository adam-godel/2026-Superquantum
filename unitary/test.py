import argparse
import json
import os
import re
import numpy as np
import scipy.linalg

from qiskit import QuantumCircuit, quantum_info
from qiskit.quantum_info import Operator
from qiskit.qasm3 import loads
from scipy.linalg import expm

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

RY = lambda theta: np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
RZ = lambda theta: np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])

def parity(x: int, idxs: tuple[int, ...]) -> int:
    p = 0
    for i in idxs:
        p ^= (x >> i) & 1
    return p

def hamiltonian_from_xor_phase_poly(n: int, poly: dict, global_phase=0.0):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)

    for x in range(dim):
        phase = global_phase
        for idxs, angle in poly.items():
            if parity(x, idxs):
                phase += angle
        H[x, x] = phase

    return H

def make_unitary(poly):
    H = hamiltonian_from_xor_phase_poly(4, poly)
    U_expm = expm(1j * H)
    return U_expm

def unitary_from_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("State vector must be non-zero.")
    state = state / norm

    basis = [state]
    dim = state.size
    for i in range(dim):
        v = np.zeros(dim, dtype=complex)
        v[i] = 1.0
        for b in basis:
            v = v - np.vdot(b, v) * b
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            basis.append(v / v_norm)
        if len(basis) == dim:
            break

    if len(basis) != dim:
        raise ValueError("Failed to construct a full orthonormal basis.")

    return np.column_stack(basis)

def _compute_challenge12():
    """U = ∏ exp(-i π k/8 P) for all terms in challenge12.json."""
    _pm = {'I': np.eye(2, dtype=complex), 'X': X, 'Y': Y, 'Z': Z}
    _path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'challenge12.json')
    with open(_path) as f:
        _data = json.load(f)
    _n   = _data['n']
    _dim = 1 << _n
    _U   = np.eye(_dim, dtype=complex)
    for _term in _data['terms']:
        # Kron in reversed order so qubit 0 is LSB (Qiskit convention)
        _P = _pm[_term['pauli'][-1]]
        for _i in range(len(_term['pauli']) - 2, -1, -1):
            _P = np.kron(_P, _pm[_term['pauli'][_i]])
        _theta = np.pi * _term['k'] / 8
        # exp(-i θ P) = cos θ · I  −  i sin θ · P   (P² = I)
        _U = (np.cos(_theta) * np.eye(_dim) - 1j * np.sin(_theta) * _P) @ _U
    return _U

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
    6: scipy.linalg.expm(1j*np.pi/7*(np.kron(X,X)+np.kron(Z,np.eye(2))+np.kron(np.eye(2),Z))),
    7: unitary_from_state(np.array([
        0.1061479384 - 0.679641467j,
        -0.3622775887 - 0.453613136j,
        0.2614190429 + 0.0445330969j,
        0.3276449279 - 0.1101628411j,
    ], dtype=complex)),
    8: np.block([
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5j, -0.5, -0.5j],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, -0.5j, -0.5, 0.5j]
    ]),
    9: np.array([
        [1, 0, 0, 0],
        [0, 0, -0.5+0.5j, 0.5+0.5j],
        [0, 1j, 0, 0],
        [0, 0, -0.5+0.5j, -0.5-0.5j]
    ]),
    10: quantum_info.random_unitary(4, seed=42).data,
    11: make_unitary({(0,): np.pi/4, (1,): np.pi/4, (2,): np.pi/4, (3,): np.pi/4,
        (0,1): np.pi/4, (0,2): np.pi/4, (0,3): np.pi/4,
        (1,2): np.pi/4, (1,3): np.pi/4, (2,3): np.pi/4,
        (1,2,3): np.pi/4}),
    12: _compute_challenge12(),
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

    for phase in np.arange(-2 * np.pi, 2 * np.pi, .0001):
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

def count_t_gates_manual(qasm_str: str) -> int:
    gate_t_counts = {}
    
    gate_defs = re.findall(r"gate\s+(\w+).*?\{(.*?)\}", qasm_str, re.DOTALL)
    
    for _ in range(5): 
        for name, body in gate_defs:
            current_count = len(re.findall(r"\b(t|tdg)\b", body))
            
            for other_name, other_count in gate_t_counts.items():
                if other_name != name:
                    calls = len(re.findall(rf"\b{other_name}\b", body))
                    current_count += (calls * other_count)
            
            gate_t_counts[name] = current_count

    main_body = re.sub(r"gate.*?\{.*?\}", "", qasm_str, flags=re.DOTALL)
    
    total_t = len(re.findall(r"\b(t|tdg)\b", main_body))
    
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
    
    if U_qasm.shape != U_expected.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"  from QASM:     {U_qasm.shape}\n"
            f"  expected dict: {U_expected.shape}\n"
            f"QASM qubits: {qc.num_qubits} -> expected dimension {2**qc.num_qubits}"
        )

    if unitary_id == 7:
        psi_expected = U_expected[:, 0]
        psi_actual   = U_qasm[:, 0]

        print("Expected state:")
        print(np.round(psi_expected, 6))
        print()

        overlap = np.vdot(psi_expected, psi_actual)
        best_phase = np.conj(overlap) / abs(overlap) if abs(overlap) > 0 else 1.0 + 0.0j
        aligned_state = best_phase * psi_actual

        print("Actual state (phase-aligned U|00⟩):")
        print(np.round(aligned_state, 6))
        print()

        fidelity = abs(overlap) ** 2
        print(f"Fidelity |⟨ψ|U|00⟩|²: {fidelity:.10f}")
    else:
        print("Expected matrix:")
        print(np.round(U_expected, 6))
        print()

        aligned = distance_global_phase(U_qasm, U_expected)

        print("Best-aligned actual matrix (phase * actual, rounded to 6 decimals):")
        print(np.round(aligned, 6))
        print()

        err = np.linalg.norm(aligned - U_expected)
        print(f"Min |Δ|: {err:.3e}")

    t_count = count_t_gates_manual(qasm_src)
    print(f"T-gate count: {t_count}")

if __name__ == "__main__":
    main()
