import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator, state_fidelity

from utils import Rz, Ry
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

basis_gates = ["h", "cx", "s", "sdg", "t", "tdg"]
max_total_gates = 10_000
target_fidelity = 0.97
epsilon_coarse = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
refine_steps = 5
num_seeds = 100

best = None
best_score = None
best_fid = None
selected_eps = None
min_total_gates = None
max_fid_overall = None

def evaluate_epsilon(eps: float) -> dict:
    candidate_best = None
    candidate_score = None
    candidate_fid = None
    candidate_min_total = None
    max_fid_eps = None

    base_qc = build_approx_circuit(eps, theta, u_angles, v_angles)
    print(f"\nEpsilon {eps:.1e}")

    for seed in range(num_seeds):
        tqc = transpile(
            base_qc,
            basis_gates=basis_gates,
            optimization_level=2,
            seed_transpiler=seed,
        )

        ops = tqc.count_ops()
        total_gates = int(sum(ops.values()))
        if candidate_min_total is None or total_gates < candidate_min_total:
            candidate_min_total = total_gates

        t_count = ops.get("t", 0) + ops.get("tdg", 0)
        depth = tqc.depth()

        if total_gates > max_total_gates:
            print(
                f"  seed={seed:02d} total={total_gates:5d} t={t_count:4d} "
                f"depth={depth:4d} fid=skipped -> skip (gate cap)"
            )
            continue

        score = (total_gates, t_count, depth)

        if candidate_score is not None and score >= candidate_score:
            print(
                f"  seed={seed:02d} total={total_gates:5d} t={t_count:4d} "
                f"depth={depth:4d} fid=skipped -> skip (worse score)"
            )
            continue

        fid = state_fidelity(Statevector.from_instruction(tqc), psi)
        if max_fid_eps is None or fid > max_fid_eps:
            max_fid_eps = fid

        if fid < target_fidelity:
            print(
                f"  seed={seed:02d} total={total_gates:5d} t={t_count:4d} "
                f"depth={depth:4d} fid={fid:.6f} -> skip (below target)"
            )
            continue

        candidate_best = tqc
        candidate_score = score
        candidate_fid = fid
        print(
            f"  seed={seed:02d} total={total_gates:5d} t={t_count:4d} "
            f"depth={depth:4d} fid={fid:.6f} -> best for epsilon"
        )

    return {
        "best": candidate_best,
        "score": candidate_score,
        "fid": candidate_fid,
        "min_total": candidate_min_total,
        "max_fid": max_fid_eps,
    }


print("\nCoarse epsilon sweep")
prev_eps = None
selected = None

for eps in epsilon_coarse:
    result = evaluate_epsilon(eps)
    if result["min_total"] is not None:
        if min_total_gates is None or result["min_total"] < min_total_gates:
            min_total_gates = result["min_total"]
    if result["max_fid"] is not None:
        if max_fid_overall is None or result["max_fid"] > max_fid_overall:
            max_fid_overall = result["max_fid"]

    if result["best"] is not None:
        selected = (eps, result)
        break
    prev_eps = eps

if selected is not None and prev_eps is not None:
    print(f"\nRefining between {prev_eps:.1e} and {selected[0]:.1e}")
    refine_eps = np.logspace(np.log10(prev_eps), np.log10(selected[0]), num=refine_steps)
    for eps in refine_eps[1:-1]:
        result = evaluate_epsilon(float(eps))
        if result["min_total"] is not None:
            if min_total_gates is None or result["min_total"] < min_total_gates:
                min_total_gates = result["min_total"]
        if result["max_fid"] is not None:
            if max_fid_overall is None or result["max_fid"] > max_fid_overall:
                max_fid_overall = result["max_fid"]

        if result["best"] is not None:
            selected = (float(eps), result)
            break

if selected is not None:
    selected_eps, selected_result = selected
    best = selected_result["best"]
    best_score = selected_result["score"]
    best_fid = selected_result["fid"]
else:
    selected_eps = None


if best is None:
    raise RuntimeError(
        "No candidate circuits met the fidelity target. "
        f"Target fidelity={target_fidelity:.3f}, "
        f"best observed fidelity={max_fid_overall}. "
        f"Smallest observed total_gates={min_total_gates}. "
        f"Increase max_total_gates (currently {max_total_gates}), "
        "decrease epsilon, or lower the target."
    )

print(f"\nSelected epsilon: {selected_eps:.1e}")
print("\nBest circuit (total gates, T-count, depth):", best_score)
print(best)
print("\nBest circuit fidelity:", best_fid)

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