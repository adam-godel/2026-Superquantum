import numpy as np
import math
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.qasm3 import dumps as dumps3

from utils import Ry
from utils import Rz
from utils import Rx
from test import count_t_gates_manual, distance_global_phase, expected as EXPECTED_DICT

def run_optimization(unitary_id, theta):
    if unitary_id not in EXPECTED_DICT:
        print(f"Error: Unitary {unitary_id} not found in test.py expected dict.")
        return
    
    expected_u = EXPECTED_DICT[unitary_id]
    epsilons = [10**(-i/2) for i in range(2, 15)] 
    
    print(f"Optimizing Unitary {unitary_id}, theta={theta:.3f}")
    print(f"{'Epsilon':<10} | {'T-Count':<8} | {'Distance':<10}")
    print("-" * 45)
    
    best_eps, best_t, best_dist = None, float('inf'), float('inf')
    
    for eps in epsilons:
        qc = QuantumCircuit(2)
        
        if unitary_id == 2:
        
            qc.append(Ry(theta/2, eps).to_gate(), [1])
            qc.cx(0, 1)
            qc.append(Ry(-theta/2, eps).to_gate(), [1])
            qc.cx(0, 1)
        
        elif unitary_id == 3:
            
            qc.cx(0, 1)
            qc.append(Rz(-2*theta, eps).to_gate(), [1])
            qc.cx(0, 1)

        elif unitary_id == 4:
            qc.h(0); qc.h(1)
            qc.s(0); qc.s(1)
            qc.h(0); qc.h(1)
            qc.cx(0, 1)
            qc.append(Rz(-2*theta, eps).to_gate(), [1])
            qc.cx(0, 1)
            qc.h(0); qc.h(1)
            qc.sdg(0); qc.sdg(1)
            qc.h(0); qc.h(1)

            qc.h(0); qc.h(1)
            qc.cx(0, 1)
            qc.append(Rz(-2*theta, eps).to_gate(), [1])
            qc.cx(0, 1)
            qc.h(0); qc.h(1)

        elif unitary_id == 6:
            qc.h(0); qc.h(1)
            qc.cx(0, 1)
            qc.append(Rz(-2*theta, eps).to_gate(), [1])
            qc.cx(0, 1)
            qc.h(0); qc.h(1)

            qc.append(Rz(-theta, eps).to_gate(), [0])
            qc.append(Rz(-theta, eps).to_gate(), [1])
        
        # Add more unitary cases as needed
        qasm_str = dumps3(qc)

        t_count = count_t_gates_manual(qasm_str)
        
        actual = Operator(qc).data
        aligned = distance_global_phase(actual, expected_u)
        dist = np.linalg.norm(aligned - expected_u)
        
        if t_count > 0 and (t_count < best_t or (t_count == best_t and dist < best_dist)):
            best_eps, best_t, best_dist = eps, t_count, dist
        
        print(f"{eps:.1e}    | {t_count:<8} | {dist:.2e}")
    
    print("-" * 45)
    if best_eps:
        print(f"Best Result: eps={best_eps:.1e}, T-count={best_t}, dist={best_dist:.2e}")
    else:
        print("No valid circuits generated. Check your Ry function.")

if __name__ == "__main__":
    # change to run w diff unitarys
    run_optimization(6, math.pi/7)


#Shared functions 

def _synthesize(axis, angle, eps, _cache):
    """Synthesize one rotation into Clifford+T.  Returns (gate, t_count)."""
    key = (axis, angle, eps)
    if key in _cache:
        return _cache[key]
    sub = {"rz": Rz, "ry": Ry, "rx": Rx}[axis](angle, eps)
    gate = sub.to_gate()
    tc = count_t_gates_manual(dumps3(sub))
    _cache[key] = (gate, tc)
    return gate, tc

def normalize_angle(a):
    """Reduce angle to (-π, π]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)

def build_circuit(ops, eps_list, _cache):
    """Assemble the full 2-qubit Clifford+T circuit."""
    qc = QuantumCircuit(2)
    rot_idx = 0
    for op in ops:
        if op[0] == "cx":
            qc.cx(op[1], op[2])
        else:
            gate, _ = _synthesize(op[0], op[2], eps_list[rot_idx], _cache)
            qc.append(gate, [op[1]])
            rot_idx += 1
    return qc

def total_t_count(ops, rotation_indices, eps_list, _cache):
    return sum(
        _synthesize(ops[idx][0], ops[idx][2], eps_list[j], _cache)[1]
        for j, idx in enumerate(rotation_indices)
    )