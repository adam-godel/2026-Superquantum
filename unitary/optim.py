import numpy as np
import math
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.qasm3 import dumps as dumps3

from utils import Ry
from utils import Rz
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

    best_eps = None
    best_t = float('inf')
    best_dist = float('inf')

    for eps in epsilons:
        qc = QuantumCircuit(2)

        if unitary_id == 2:
            qc.append(Ry(theta/2, eps).to_gate(), [0]) 
            qc.cx(1, 0)
            qc.append(Ry(-theta/2, eps).to_gate(), [0])
            qc.cx(1, 0)

        elif unitary_id == 3:
            qc.cx(0, 1)
            qc.append(Rz(-2 * theta, eps).to_gate(), [1])
            qc.cx(0, 1)

        elif unitary_id == 4:
            qc.h(0); qc.h(1)
            qc.s(0); qc.s(1)
            qc.h(0); qc.h(1)
            qc.cx(0, 1)
            qc.append(Rz(-2 * theta, eps).to_gate(), [1])
            qc.cx(0, 1)
            qc.h(0); qc.h(1)
            qc.sdg(0); qc.sdg(1)
            qc.h(0); qc.h(1)

            qc.h(0); qc.h(1)
            qc.cx(0, 1)
            qc.append(Rz(-2 * theta, eps).to_gate(), [1])
            qc.cx(0, 1)
            qc.h(0); qc.h(1)

        elif unitary_id == 6:
            qc.h(0); qc.h(1)
            qc.cx(0, 1)
            qc.append(Rz(-2 * theta, eps).to_gate(), [1])
            qc.cx(0, 1)
            qc.h(0); qc.h(1)

            qc.append(Rz(-theta, eps).to_gate(), [0])
            qc.append(Rz(-theta, eps).to_gate(), [1])

        qasm_str = dumps3(qc)
        t_count = count_t_gates_manual(qasm_str)

        actual = Operator(qc).data
        aligned = distance_global_phase(actual, expected_u)
        dist = np.linalg.norm(aligned - expected_u)

        if (
            t_count < best_t or
            (t_count == best_t and dist < best_dist)
        ):
            best_eps = eps
            best_t = t_count
            best_dist = dist

        print(f"{eps:.1e}    | {t_count:<8} | {dist:.2e}")

    print("-" * 45)
    print(f"Best Result: eps={best_eps:.1e}, T-count={best_t}, dist={best_dist:.2e}")



if __name__ == "__main__":
    # change to run w diff unitarys
    run_optimization(2, math.pi/7)