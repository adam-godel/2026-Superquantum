import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator, state_fidelity
from qiskit.circuit.library import UnitaryGate
from qiskit.qasm3 import dumps as dumps3


# ============================================================
# 1. Define target state |psi> in computational basis order
#    |00>, |01>, |10>, |11>
# ============================================================

psi = np.array([
    0.1061479384 - 0.6796414670j,   # |00>
    -0.3622775887 - 0.4536131360j,   # |01>
    0.2614190429 + 0.0445330969j,   # |10>
    0.3276449279 - 0.1101628411j    # |11>
], dtype=complex)

psi = psi / np.linalg.norm(psi)    # normalize (safe)


# ============================================================
# 2. Schmidt decomposition (q1 | q0)
# ============================================================

# Reshape amplitudes into a 2x2 matrix
M = psi.reshape(2, 2)

U, s, Vh = np.linalg.svd(M)
V = Vh.conj().T

s0, s1 = s
theta = 2 * np.arctan2(s1, s0)

V_star = V.conj()


# ============================================================
# 3. Build the unitary-only state-preparation circuit
# ============================================================

qc = QuantumCircuit(2)

# Prepare Schmidt form: s0|00> + s1|11>
qc.ry(theta, 1)
qc.cx(1, 0)

# Local unitaries
qc.append(UnitaryGate(U), [1])
qc.append(UnitaryGate(V_star), [0])


# ============================================================
# 4. Verify correctness (fidelity ≈ 1)
# ============================================================

sv = Statevector.from_instruction(qc)
fid = state_fidelity(sv, psi)

print("\nInitial circuit fidelity:", fid)


# ============================================================
# 5. Optimize for T-count (Clifford+T basis)
# ============================================================

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


# ============================================================
# 6. Extract the final 4x4 unitary matrix
# ============================================================

U4 = Operator(best).data

print("\nFinal 4x4 unitary matrix U:\n")
np.set_printoptions(precision=6, suppress=True)
print(U4)


# ============================================================
# 7. Sanity check: U|00> = |psi|
# ============================================================

e00 = np.array([1, 0, 0, 0], dtype=complex)
out = U4 @ e00

print("\nMax |U|00> - psi| =", np.max(np.abs(out - psi)))
