import numpy as np
from qiskit import QuantumCircuit, quantum_info, transpile
from qiskit.quantum_info import Operator
from qiskit.qasm3 import dumps as dumps3
from qiskit.circuit.library import UnitaryGate

from utils import Rz, Ry, Rx
from test import count_t_gates_manual


# ============================================================
# 1. Target unitary & KAK template via transpile
# ============================================================
# transpile with basis_gates=["u3","cx"] performs a KAK (Cartan) decomposition
# of the 4x4 unitary into single-qubit u3 gates and entangling CX gates.
# A generic 2-qubit unitary decomposes into at most 3 CX + 6 u3 gates.
unitary = quantum_info.random_unitary(4, seed=42)
target = unitary.data

template_qc = QuantumCircuit(2)
template_qc.append(UnitaryGate(target), [0, 1])
template = transpile(
    template_qc,
    basis_gates=["u3", "cx"],
    optimization_level=2,
    seed_transpiler=0,
)


# ============================================================
# 2. Parse template into a flat ops list
# ============================================================
# Each u3(θ, φ, λ) gate is equivalent to Rz(φ)·Ry(θ)·Rz(λ) up to a
# per-gate global phase of e^{i(φ+λ)/2}.  These per-gate phases multiply
# into an overall global phase on the full circuit, which is unobservable
# and drops out of our phase-invariant distance metric.
ANGLE_TOL = 1e-9


def normalize_angle(a):
    """Reduce angle to (-π, π]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


# ops entries:
#   ("cx",  ctrl_qubit, tgt_qubit)
#   ("rz",  qubit, angle)   ┐
#   ("ry",  qubit, angle)   ├ rotation gates requiring Clifford+T synthesis
#   ("rx",  qubit, angle)   ┘
ops = []

for inst in template.data:
    name = inst.operation.name
    qubits = [template.find_bit(q).index for q in inst.qubits]

    if name == "cx":
        ops.append(("cx", qubits[0], qubits[1]))

    elif name in ("u3", "u"):
        theta, phi, lam = [float(p) for p in inst.operation.params]
        q = qubits[0]
        for axis, angle in [("rz", lam), ("ry", theta), ("rz", phi)]:
            a = normalize_angle(angle)
            if abs(a) > ANGLE_TOL:
                ops.append((axis, q, a))

    elif name in ("rz", "ry", "rx"):
        a = normalize_angle(float(inst.operation.params[0]))
        if abs(a) > ANGLE_TOL:
            ops.append((name, qubits[0], a))

    elif name == "p":
        # Phase gate p(λ) = diag(1, e^{iλ}) = e^{iλ/2} · Rz(λ)
        a = normalize_angle(float(inst.operation.params[0]))
        if abs(a) > ANGLE_TOL:
            ops.append(("rz", qubits[0], a))

    elif name in ("id", "barrier"):
        pass

    else:
        raise ValueError(f"Unexpected gate in KAK template: {name}")

# Indices into ops[] that are rotation gates needing synthesis
rotation_indices = [i for i, op in enumerate(ops) if op[0] in ("rx", "ry", "rz")]
n_rotations = len(rotation_indices)


# ============================================================
# 3. Synthesis cache & circuit builder
# ============================================================
# Each rotation is synthesised independently via gridsynth.  Results are
# cached so that the same (axis, angle, epsilon) triple is never computed
# twice — critical for the greedy Phase 2 loop.
_cache: dict[tuple, tuple] = {}  # (axis, angle, eps) -> (Gate, t_count)


def _synthesize(axis, angle, eps):
    """Synthesize one rotation gate into Clifford+T.  Returns (gate, t_count)."""
    key = (axis, angle, eps)
    if key in _cache:
        return _cache[key]
    if axis == "rz":
        subcircuit = Rz(angle, eps)
    elif axis == "ry":
        subcircuit = Ry(angle, eps)
    else:
        subcircuit = Rx(angle, eps)
    gate = subcircuit.to_gate()
    tc = count_t_gates_manual(dumps3(subcircuit))
    _cache[key] = (gate, tc)
    return gate, tc


def build_circuit(eps_list):
    """Assemble the full 2-qubit Clifford+T circuit from per-rotation epsilons."""
    qc = QuantumCircuit(2)
    rot_idx = 0
    for op in ops:
        tag = op[0]
        if tag == "cx":
            qc.cx(op[1], op[2])
        else:  # rx / ry / rz
            gate, _ = _synthesize(tag, op[2], eps_list[rot_idx])
            qc.append(gate, [op[1]])
            rot_idx += 1
    return qc


def total_t_count(eps_list):
    """Total T-gate count: sum over all rotation subcircuits.

    CX and Clifford gates contribute zero T gates, so the per-rotation
    sum equals the full-circuit T-count exactly.
    """
    total = 0
    for j, idx in enumerate(rotation_indices):
        axis = ops[idx][0]
        angle = ops[idx][2]
        _, tc = _synthesize(axis, angle, eps_list[j])
        total += tc
    return total


# ============================================================
# 4. Phase-invariant operator distance
# ============================================================
def operator_distance(actual, reference):
    """Frobenius distance minimised over global phase.

    For two d×d unitaries A, B:
        min_φ  ‖e^{iφ} A − B‖_F  =  sqrt(2d − 2|tr(A†B)|)

    Proof: expand the squared norm, optimise over φ analytically.
    """
    d = actual.shape[0]
    inner = np.trace(actual.conj().T @ reference)
    return float(np.sqrt(max(2 * d - 2 * np.abs(inner), 0.0)))


# ============================================================
# 5. Phase 1 — Coarse uniform epsilon sweep
# ============================================================
# Sweep epsilon from coarse (few T gates, low accuracy) to fine
# (many T gates, high accuracy).  Stop at the first epsilon whose
# distance falls below TARGET_DIST — that is the coarsest (cheapest)
# epsilon that meets the accuracy requirement.
eps_coarse = [
    1e-1, 5e-2, 2e-2, 1e-2,
    5e-3, 2e-3, 1e-3,
    5e-4, 2e-4, 1e-4,
    5e-5, 2e-5, 1e-5,
]
TARGET_DIST = 0.05  # Frobenius distance target

print(f"Rotation gates extracted from KAK template: {n_rotations}")
for j, idx in enumerate(rotation_indices):
    print(f"  rot[{j}]: {ops[idx][0]}({ops[idx][2]:+.6f}) on q{ops[idx][1]}")

print(f"\n=== Phase 1: Uniform epsilon sweep ===")
print(f"{'eps':<10} {'T-count':<10} {'distance':<12} note")
print("-" * 52)

phase1_hit = False  # whether we found an eps meeting TARGET_DIST

for eps in eps_coarse:
    eps_list = [eps] * n_rotations
    tc = total_t_count(eps_list)

    qc = build_circuit(eps_list)
    dist = operator_distance(Operator(qc).data, target)

    note = "<-- target met" if dist < TARGET_DIST else ""
    print(f"{eps:<10.1e} {tc:<10d} {dist:<12.6f} {note}")

    if dist < TARGET_DIST:
        phase1_hit = True
        break  # coarsest passing epsilon found

# Determine starting point for Phase 2
if phase1_hit:
    best_uniform_eps = eps
else:
    best_uniform_eps = eps_coarse[-1]
    print(f"\nWARNING: target distance {TARGET_DIST} not reached at any "
          f"coarse epsilon.  Starting Phase 2 from eps={best_uniform_eps:.1e}")

current_eps = [best_uniform_eps] * n_rotations
current_t = total_t_count(current_eps)
current_dist = operator_distance(
    Operator(build_circuit(current_eps)).data, target
)
print(f"\nPhase 1 → starting point: eps={best_uniform_eps:.1e}, "
      f"T={current_t}, dist={current_dist:.6f}")


# ============================================================
# 6. Phase 2 — Per-rotation greedy relaxation
# ============================================================
# Each rotation can be independently loosened (larger epsilon = fewer T
# gates) as long as the overall operator distance stays below TARGET_DIST.
#
# Strategy: for each rotation, try increasingly aggressive relaxation
# factors (largest first).  Accept the first (most aggressive) factor
# that keeps the distance in budget.  Repeat passes over all rotations
# until no rotation can be further relaxed.
#
# This works because T-count is monotonically non-increasing with epsilon
# for a fixed angle (looser tolerance ⇒ gridsynth uses fewer T gates).
RELAXATION_FACTORS = [100, 50, 20, 10, 5, 2]

print(f"\n=== Phase 2: Per-rotation greedy relaxation ===")

iteration = 0
while True:
    iteration += 1
    any_improved = False
    print(f"\n  --- Iteration {iteration} "
          f"(T={current_t}, dist={current_dist:.6f}) ---")

    for j in range(n_rotations):
        idx = rotation_indices[j]
        axis  = ops[idx][0]
        angle = ops[idx][2]
        orig_eps = current_eps[j]
        _, orig_tc_j = _synthesize(axis, angle, orig_eps)

        for factor in RELAXATION_FACTORS:
            trial_eps = orig_eps * factor
            if trial_eps > 0.5:
                continue  # don't go absurdly loose

            _, trial_tc_j = _synthesize(axis, angle, trial_eps)
            if trial_tc_j >= orig_tc_j:
                continue  # gridsynth found no shorter sequence

            # T-count would improve — check distance constraint
            trial_eps_list = current_eps.copy()
            trial_eps_list[j] = trial_eps
            trial_dist = operator_distance(
                Operator(build_circuit(trial_eps_list)).data, target
            )

            if trial_dist < TARGET_DIST:
                new_t = current_t - orig_tc_j + trial_tc_j
                print(f"    rot[{j}] {axis}({angle:+.4f}) q{ops[idx][1]}: "
                      f"eps {orig_eps:.1e} → {trial_eps:.1e}, "
                      f"T {current_t} → {new_t}, dist {trial_dist:.6f}")
                current_eps[j] = trial_eps
                current_t   = new_t
                current_dist = trial_dist
                any_improved = True
                break  # took most-aggressive valid relaxation; next rotation

    if not any_improved:
        print("  No further relaxation possible.")
        break


# ============================================================
# 7. Final circuit & QASM output
# ============================================================
qc = build_circuit(current_eps)
final_t = total_t_count(current_eps)
final_dist = operator_distance(Operator(qc).data, target)

qasm3_str = dumps3(qc)

# Verify T-count independently via full-QASM nested-gate parsing
verified_t = count_t_gates_manual(qasm3_str)

print(f"\n{'=' * 52}")
print(f"  FINAL RESULT")
print(f"{'=' * 52}")
print(f"  T-count  (sum):    {final_t}")
print(f"  T-count  (QASM):   {verified_t}")
print(f"  Distance:          {final_dist:.6e}")
print(f"  Per-rotation breakdown:")
for j, idx in enumerate(rotation_indices):
    axis  = ops[idx][0]
    angle = ops[idx][2]
    _, tc_j = _synthesize(axis, angle, current_eps[j])
    print(f"    rot[{j}] {axis}({angle:+.4f}) q{ops[idx][1]}: "
          f"eps={current_eps[j]:.1e}  T={tc_j}")

with open("qasm/unitary10.qasm", "w") as file:
    file.write(qasm3_str)
print(f"\nSaved to qasm/unitary10.qasm")
