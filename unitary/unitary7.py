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


def decompose_zyz(U_in: np.ndarray) -> tuple[float, float, float, float]:
    det = np.linalg.det(U_in)
    phase = np.angle(det) / 2
    U_local = U_in * np.exp(-1j * phase)

    a = U_local[0, 0]
    b = U_local[0, 1]
    c = U_local[1, 0]

    theta_local = 2 * np.arctan2(np.abs(b), np.abs(a))

    if np.isclose(np.abs(b), 0.0, atol=1e-12):
        phi_local = 0.0
        lam_local = -2 * np.angle(a)
    elif np.isclose(np.abs(a), 0.0, atol=1e-12):
        phi_local = 2 * np.angle(c)
        lam_local = 0.0
    else:
        phi_minus_lambda = 2 * np.angle(c)
        phi_plus_lambda = -2 * np.angle(a)
        phi_local = 0.5 * (phi_minus_lambda + phi_plus_lambda)
        lam_local = 0.5 * (phi_plus_lambda - phi_minus_lambda)

    return phase, phi_local, theta_local, lam_local


def append_zyz(qc: QuantumCircuit, qubit: int, phi: float, theta: float, lam: float, eps: float) -> None:
    if not np.isclose(phi, 0.0, atol=1e-12):
        qc.append(Rz(phi, eps).to_gate(), [qubit])
    if not np.isclose(theta, 0.0, atol=1e-12):
        qc.append(Ry(theta, eps).to_gate(), [qubit])
    if not np.isclose(lam, 0.0, atol=1e-12):
        qc.append(Rz(lam, eps).to_gate(), [qubit])


def build_approx_circuit(
    eps: float,
    theta_in: float,
    u_angles: tuple[float, float, float],
    v_angles: tuple[float, float, float],
) -> QuantumCircuit:
    qc_local = QuantumCircuit(2)

    qc_local.append(Ry(theta_in, eps).to_gate(), [1])
    qc_local.cx(1, 0)

    append_zyz(qc_local, 1, *u_angles, eps=eps)
    append_zyz(qc_local, 0, *v_angles, eps=eps)

    return qc_local


# ============================================================
# 3. Build a ZYZ-based approximation (no UnitaryGate)
# ============================================================

_, u_phi, u_theta, u_lam = decompose_zyz(U)
_, v_phi, v_theta, v_lam = decompose_zyz(V_star)
u_angles = (u_phi, u_theta, u_lam)
v_angles = (v_phi, v_theta, v_lam)


# ============================================================
# 4. Search epsilons for fidelity >= target
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
