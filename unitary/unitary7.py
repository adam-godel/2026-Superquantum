"""Clifford+T state-preparation synthesis for a 2-qubit state.

Strategy
--------
1. Generate multiple 4×4 unitaries whose first column is the target state,
   parameterized as  base @ diag(1, V)  with  V ∈ U(3).  This covers the
   full space of orthonormal completions; different completions yield
   different KAK rotation angles, so some are intrinsically cheaper.

2. For each candidate unitary, decompose via KAK → u3+cx, synthesize every
   rotation with gridsynth, and greedily relax individual epsilons while
   the state-preparation fidelity holds.

3. Pick the candidate with the fewest total T gates.

Key insight
-----------
Only U|00⟩ matters, so we use fidelity
    F = |⟨ψ| U |00⟩|²       (1 = perfect, 0 = worst)
instead of full operator Frobenius distance.  This is strictly weaker than
the operator metric in unitary10.py, giving the synthesizer more room to
trade accuracy on the unused columns for fewer T gates.
"""

import numpy as np
from qiskit import QuantumCircuit, quantum_info, transpile
from qiskit.quantum_info import Operator
from qiskit.qasm3 import dumps as dumps3
from qiskit.circuit.library import UnitaryGate

from utils import Rz, Ry, Rx
from test import count_t_gates_manual

# ── target (must match test.py case 7) ─────────────────────────────────────
statevector = quantum_info.random_statevector(4, seed=42).data

# ── tuning knobs ───────────────────────────────────────────────────────────
N_CANDIDATES       = 20
CANDIDATE_SEED     = 42
TARGET_FIDELITY    = 0.9975     # minimum acceptable |⟨ψ|U|00⟩|²
ANGLE_TOL          = 1e-9

EPS_COARSE = [
    1e-1, 5e-2, 2e-2, 1e-2,
    5e-3, 2e-3, 1e-3,
    5e-4, 2e-4, 1e-4,
    5e-5, 2e-5, 1e-5,
]
RELAXATION_FACTORS = [100, 50, 20, 10, 5, 2]


# ── candidate generation ───────────────────────────────────────────────────

def _gram_schmidt_completion(sv):
    """Extend sv to a 4×4 unitary (sv = column 0) via Gram-Schmidt.

    Algorithm identical to test.py's unitary_from_state, so candidate 0
    matches the test harness exactly.
    """
    sv = np.asarray(sv, dtype=complex).flatten()
    sv = sv / np.linalg.norm(sv)
    basis = [sv]
    for i in range(4):
        v = np.zeros(4, dtype=complex)
        v[i] = 1.0
        for b in basis:
            v -= np.vdot(b, v) * b
        n = np.linalg.norm(v)
        if n > 1e-12:
            basis.append(v / n)
        if len(basis) == 4:
            break
    return np.column_stack(basis)


def generate_candidates(sv, n, seed):
    """Return n unitaries that each map |00⟩ → sv.

    Candidate 0 : Gram-Schmidt completion (deterministic, matches test.py).
    Candidates 1…n-1 : base @ diag(1, V) with V drawn Haar-uniformly from
                        U(3), spanning every possible orthonormal completion.
    """
    base = _gram_schmidt_completion(sv)
    candidates = [base]
    rng = np.random.default_rng(seed)
    for _ in range(n - 1):
        # Haar-random U(3): QR of a Ginibre matrix, phase-corrected
        A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(A)
        Q = Q @ np.diag(R.diagonal() / np.abs(R.diagonal()))
        D = np.eye(4, dtype=complex)
        D[1:, 1:] = Q
        candidates.append(base @ D)
    return candidates


# ── fidelity metric ────────────────────────────────────────────────────────

def state_prep_fidelity(unitary_matrix, target_sv):
    """F = |⟨ψ| U |00⟩|²   ∈ [0, 1],  phase-invariant.

    Only the first column of U is used, so this is strictly weaker than
    comparing full operator matrices and allows coarser per-rotation epsilons.
    """
    overlap = np.vdot(target_sv, unitary_matrix[:, 0])   # ⟨ψ|U|0⟩
    return float(np.abs(overlap) ** 2)


# ── gate extraction (KAK → u3 + cx → Rz/Ry/Rx list) ──────────────────────

def normalize_angle(a):
    """Reduce angle to (-π, π]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def extract_ops(target_matrix):
    """Transpile a 2-qubit unitary and return a flat op list.

    Each entry is one of:
        ("cx",  ctrl, tgt)
        ("rz",  qubit, angle)  /  ("ry", ...)  /  ("rx", ...)
    """
    qc = QuantumCircuit(2)
    qc.append(UnitaryGate(target_matrix), [0, 1])
    template = transpile(
        qc, basis_gates=["u3", "cx"],
        optimization_level=2, seed_transpiler=0,
    )
    ops = []
    for inst in template.data:
        name   = inst.operation.name
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
            a = normalize_angle(float(inst.operation.params[0]))
            if abs(a) > ANGLE_TOL:
                ops.append(("rz", qubits[0], a))

        elif name in ("id", "barrier"):
            pass

        else:
            raise ValueError(f"Unexpected gate: {name}")

    return ops


# ── Clifford+T synthesis helpers ───────────────────────────────────────────

_cache: dict[tuple, tuple] = {}


def _synthesize(axis, angle, eps):
    """Synthesize one rotation into Clifford+T.  Returns (gate, t_count)."""
    key = (axis, angle, eps)
    if key in _cache:
        return _cache[key]
    sub = {"rz": Rz, "ry": Ry, "rx": Rx}[axis](angle, eps)
    gate = sub.to_gate()
    tc   = count_t_gates_manual(dumps3(sub))
    _cache[key] = (gate, tc)
    return gate, tc


def build_circuit(ops, eps_list):
    """Assemble the full 2-qubit Clifford+T circuit."""
    qc = QuantumCircuit(2)
    rot_idx = 0
    for op in ops:
        if op[0] == "cx":
            qc.cx(op[1], op[2])
        else:
            gate, _ = _synthesize(op[0], op[2], eps_list[rot_idx])
            qc.append(gate, [op[1]])
            rot_idx += 1
    return qc


def total_t_count(ops, rotation_indices, eps_list):
    """Sum T-counts over all synthesized rotations."""
    return sum(
        _synthesize(ops[idx][0], ops[idx][2], eps_list[j])[1]
        for j, idx in enumerate(rotation_indices)
    )


# ── per-candidate optimization (Phase 1 + Phase 2) ────────────────────────

def optimize_candidate(target_matrix, target_sv, cid):
    """Phase 1 (eps sweep) + Phase 2 (greedy relax) for one candidate.

    Returns (qc, t_count, fidelity, ops, rotation_indices, eps_list)
    or None if the target fidelity is never reached.
    """
    ops              = extract_ops(target_matrix)
    rotation_indices = [i for i, op in enumerate(ops) if op[0] in ("rx", "ry", "rz")]
    n_rot            = len(rotation_indices)
    n_cx             = sum(1 for op in ops if op[0] == "cx")

    # Phase 1: find the coarsest uniform eps that meets the fidelity threshold
    hit_eps = None
    for eps in EPS_COARSE:
        eps_list = [eps] * n_rot
        qc   = build_circuit(ops, eps_list)
        fid  = state_prep_fidelity(Operator(qc).data, target_sv)
        if fid >= TARGET_FIDELITY:
            hit_eps = eps
            break

    if hit_eps is None:
        print(f"  [{cid:2d}] SKIP – fidelity threshold not reached  "
              f"({n_rot} rot, {n_cx} cx)")
        return None

    current_eps = [hit_eps] * n_rot
    current_t   = total_t_count(ops, rotation_indices, current_eps)
    current_fid = state_prep_fidelity(
        Operator(build_circuit(ops, current_eps)).data,
        target_sv,
    )

    # Phase 2: greedily relax individual rotations
    for _ in range(20):                          # max outer passes
        improved = False
        for j in range(n_rot):
            idx       = rotation_indices[j]
            axis      = ops[idx][0]
            angle     = ops[idx][2]
            orig_eps  = current_eps[j]
            _, orig_tc = _synthesize(axis, angle, orig_eps)

            for factor in RELAXATION_FACTORS:
                trial_eps = orig_eps * factor
                if trial_eps > 0.5:
                    continue
                _, trial_tc = _synthesize(axis, angle, trial_eps)
                if trial_tc >= orig_tc:
                    continue                      # no T saving → skip

                trial_eps_list = current_eps.copy()
                trial_eps_list[j] = trial_eps
                trial_fid = state_prep_fidelity(
                    Operator(build_circuit(ops, trial_eps_list)).data,
                    target_sv,
                )

                if trial_fid >= TARGET_FIDELITY:
                    current_eps[j]  = trial_eps
                    current_t      += trial_tc - orig_tc
                    current_fid     = trial_fid
                    improved        = True
                    break             # move to next rotation

        if not improved:
            break

    qc = build_circuit(ops, current_eps)
    print(f"  [{cid:2d}] T={current_t:4d}  fidelity={current_fid:.10f}  "
          f"({n_rot} rot, {n_cx} cx)")
    return qc, current_t, current_fid, ops, rotation_indices, current_eps


# ── main ───────────────────────────────────────────────────────────────────

candidates = generate_candidates(statevector, N_CANDIDATES, CANDIDATE_SEED)

print(f"Statevector:      {np.round(statevector, 6)}")
print(f"Candidates:       {N_CANDIDATES}  |  fidelity threshold: {TARGET_FIDELITY}")
print("=" * 60)

best = None   # (t_count, fidelity, qc, cid, ops, rotation_indices, eps_list)

for i, cand in enumerate(candidates):
    result = optimize_candidate(cand, statevector, i)
    if result is None:
        continue
    qc, tc, fid, ops, rot_idx, eps_list = result
    if best is None or tc < best[0] or (tc == best[0] and fid > best[1]):
        best = (tc, fid, qc, i, ops, rot_idx, eps_list)

if best is None:
    print("\nERROR: no candidate met the fidelity target.")
else:
    tc, fid, qc, cid, ops, rot_idx, eps_list = best

    qasm3_str  = dumps3(qc)
    verified_t = count_t_gates_manual(qasm3_str)

    print(f"\n{'=' * 52}")
    print(f"  FINAL RESULT  (candidate {cid})")
    print(f"{'=' * 52}")
    print(f"  T-count  (sum):         {tc}")
    print(f"  T-count  (QASM):        {verified_t}")
    print(f"  Fidelity |⟨ψ|U|00⟩|²:  {fid:.10f}")
    print(f"  Per-rotation breakdown:")
    for j, idx in enumerate(rot_idx):
        axis  = ops[idx][0]
        angle = ops[idx][2]
        _, tc_j = _synthesize(axis, angle, eps_list[j])
        print(f"    rot[{j}] {axis}({angle:+.6f}) q{ops[idx][1]}: "
              f"eps={eps_list[j]:.1e}  T={tc_j}")

    with open("qasm/unitary7.qasm", "w") as f:
        f.write(qasm3_str)
    print(f"\n  Saved to qasm/unitary7.qasm")
