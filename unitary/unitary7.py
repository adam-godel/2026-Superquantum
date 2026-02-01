import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator, state_fidelity
from qiskit.circuit.library import UnitaryGate
from qiskit.qasm3 import dumps as dumps3

psi = np.array([
    0.1061479384 - 0.6796414670j,   # |00>
    -0.3622775887 - 0.4536131360j,   # |01>
    0.2614190429 + 0.0445330969j,   # |10>
    0.3276449279 - 0.1101628411j    # |11>
], dtype=complex)

psi = psi / np.linalg.norm(psi)    # normalize (safe)
M = psi.reshape(2, 2)
U, s, Vh = np.linalg.svd(M)
V = Vh.conj().T
s0, s1 = s
theta = 2 * np.arctan2(s1, s0)
V_star = V.conj()

qc = QuantumCircuit(2)

qc.ry(theta, 1)
qc.cx(1, 0)

qc.append(UnitaryGate(U), [1])
qc.append(UnitaryGate(V_star), [0])

sv = Statevector.from_instruction(qc)
fid = state_fidelity(sv, psi)

print("\nInitial circuit fidelity:", fid)

basis_gates = ["rz", "sx", "x", "cx"]

best = None
best_score = None

for seed in range(100):
    tqc = transpile(
        qc,
        basis_gates=basis_gates,
        optimization_level=3,
        seed_transpiler=seed,
    )

    ops = tqc.count_ops()
    t_count = ops.get("t", 0) + ops.get("tdg", 0)
    depth = tqc.depth()

    score = (t_count, depth)

    if best_score is None or score < best_score:
        best = tqc
        best_score = score


print("\nBest circuit (T-count, depth):", best_score)
print(best)

qasm3_str = dumps3(best)
with open("qasm/unitary7.qasm", "w") as file:
    file.write(qasm3_str)

U4 = Operator(best).data

print("\nFinal 4x4 unitary matrix U:\n")
np.set_printoptions(precision=6, suppress=True)
print(U4)

e00 = np.array([1, 0, 0, 0], dtype=complex)
out = U4 @ e00

print("\nMax |U|00> - psi| =", np.max(np.abs(out - psi)))