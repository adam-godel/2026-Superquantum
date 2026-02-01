import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import argparse
import numpy as np
import multiprocessing as mp

from qiskit import QuantumCircuit, quantum_info, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit.qasm3 import dumps as dumps3
from qiskit.circuit.library import UnitaryGate

from utils import Rz, Ry, Rx
from test import count_t_gates_manual

statevector = quantum_info.random_statevector(4, seed=42).data
_target_sv = np.asarray(statevector, dtype=complex).flatten()
_target_sv_conj = np.conj(_target_sv)

N_CANDIDATES       = 200
CANDIDATE_SEED     = 42
TARGET_FIDELITY    = 0.999999
ANGLE_TOL          = 1e-9

EPS_COARSE = [10**(-i/2) for i in range(2, 18)]
RELAXATION_FACTORS = [100, 50, 30, 20, 15, 10, 7, 5, 3, 2, 1.5, 1.3, 1.2, 1.1, 1.05, 1.02]

FIDELITY_MARGIN = 5e-11

_SV0 = Statevector.from_int(0, 4)

_CX_PERM = {
    (0, 1): np.array([0, 3, 2, 1], dtype=np.int64),
    (1, 0): np.array([0, 1, 3, 2], dtype=np.int64),
}

def _gram_schmidt_completion(sv):
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
    base = _gram_schmidt_completion(sv)
    candidates = [base]
    rng = np.random.default_rng(seed)
    for _ in range(n - 1):
        A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(A)
        Q = Q @ np.diag(R.diagonal() / np.abs(R.diagonal()))
        D = np.eye(4, dtype=complex)
        D[1:, 1:] = Q
        candidates.append(base @ D)
    return candidates

def normalize_angle(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def extract_ops(target_matrix):
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

_cache = {}  # key -> {"gate": Gate, "t_count": int, "mat": np.ndarray}


def synthesize(axis, angle, eps):
    """Synthesize one rotation into Clifford+T. Returns (gate, t_count)."""
    key = (axis, float(angle), float(eps))
    if key in _cache and _cache[key].get("gate") is not None:
        e = _cache[key]
        return e["gate"], e["t_count"]
    sub = {"rz": Rz, "ry": Ry, "rx": Rx}[axis](float(angle), float(eps))
    gate = sub.to_gate()
    tc   = count_t_gates_manual(dumps3(sub))
    _cache[key] = {"gate": gate, "t_count": tc, "mat": None}
    return gate, tc


def gate_matrix(axis, angle, eps):
    key = (axis, float(angle), float(eps))
    if key not in _cache or _cache[key].get("gate") is None:
        synthesize(axis, angle, eps)
    e = _cache[key]
    if e["mat"] is None:
        e["mat"] = Operator(e["gate"]).data
    return e["mat"]


def apply_1q_gate_batch(state_b4, U2, qubit):
    """
    state_b4: (B,4)
    U2: (2,2) or (B,2,2)
    """
    B = state_b4.shape[0]
    st = state_b4.reshape(B, 2, 2)
    if qubit == 0:
        if U2.ndim == 2:
            out = np.einsum("bij,kj->bik", st, U2)
        else:
            out = np.einsum("bij,bkj->bik", st, U2)
    elif qubit == 1:
        if U2.ndim == 2:
            out = np.einsum("ij,bjk->bik", U2, st)
        else:
            out = np.einsum("bij,bjk->bik", U2, st)
    else:
        raise ValueError("qubit must be 0 or 1")
    return out.reshape(B, 4)


def apply_cx_batch(state_b4, control, target):
    perm = _CX_PERM[(control, target)]
    return state_b4[:, perm]


def simulate_batch(ops, eps_matrix):
    """
    eps_matrix: (B, n_rot) float
    returns: (B,4) complex statevectors
    """
    B = eps_matrix.shape[0]
    state = np.zeros((B, 4), dtype=np.complex128)
    state[:, 0] = 1.0

    rot_idx = 0
    for op in ops:
        if op[0] == "cx":
            state = apply_cx_batch(state, op[1], op[2])
        else:
            axis, q, angle = op[0], op[1], op[2]
            eps_vals = eps_matrix[:, rot_idx]
            if np.all(eps_vals == eps_vals[0]):
                U = gate_matrix(axis, angle, float(eps_vals[0]))
            else:
                U = np.stack([gate_matrix(axis, angle, float(eps)) for eps in eps_vals], axis=0)
            state = apply_1q_gate_batch(state, U, q)
            rot_idx += 1

    return state


def fidelities_from_states(states_b4):
    overlaps = np.sum(_target_sv_conj[None, :] * states_b4, axis=1)  # <psi|out>
    return np.abs(overlaps) ** 2


def fidelity_exact_qiskit(qc):
    out = _SV0.evolve(qc).data
    overlap = np.vdot(_target_sv, out)
    return float(np.abs(overlap) ** 2)

def build_circuit(ops, eps_list):
    """Assemble the full 2-qubit Clifford+T circuit."""
    qc = QuantumCircuit(2)
    rot_idx = 0
    for op in ops:
        if op[0] == "cx":
            qc.cx(op[1], op[2])
        else:
            gate, _ = synthesize(op[0], op[2], eps_list[rot_idx])
            qc.append(gate, [op[1]])
            rot_idx += 1
    return qc


def total_t_count(ops, rotation_indices, eps_list):
    """Sum T-counts over all synthesized rotations."""
    return sum(
        synthesize(ops[idx][0], ops[idx][2], eps_list[j])[1]
        for j, idx in enumerate(rotation_indices)
    )

def optimize_candidate_ops(ops, cid):
    rotation_indices = [i for i, op in enumerate(ops) if op[0] in ("rx", "ry", "rz")]
    n_rot = len(rotation_indices)

    eps_vals = np.array([float(e) for e in EPS_COARSE], dtype=np.float64)
    eps_matrix = np.tile(eps_vals[:, None], (1, n_rot))
    states = simulate_batch(ops, eps_matrix)
    fids = fidelities_from_states(states)

    hit_eps = None
    for eps, fid in zip(eps_vals.tolist(), fids.tolist()):
        if fid >= TARGET_FIDELITY + FIDELITY_MARGIN:
            hit_eps = eps
            break
        if fid >= TARGET_FIDELITY - FIDELITY_MARGIN:
            qc_tmp = build_circuit(ops, [eps] * n_rot)
            if fidelity_exact_qiskit(qc_tmp) >= TARGET_FIDELITY:
                hit_eps = eps
                break

    if hit_eps is None:
        return (cid, None)

    current_eps = [float(hit_eps)] * n_rot
    current_t   = total_t_count(ops, rotation_indices, current_eps)

    max_iterations = 50
    for _ in range(max_iterations):
        improved = False
        for j in range(n_rot):
            idx        = rotation_indices[j]
            axis       = ops[idx][0]
            angle      = ops[idx][2]
            orig_eps   = float(current_eps[j])
            _, orig_tc = synthesize(axis, angle, orig_eps)

            trial_eps_list = []
            trial_tc_list  = []
            for factor in RELAXATION_FACTORS:
                trial_eps = orig_eps * float(factor)
                if trial_eps > 0.5:
                    continue
                _, trial_tc = synthesize(axis, angle, trial_eps)
                if trial_tc >= orig_tc:
                    continue
                trial_eps_list.append(float(trial_eps))
                trial_tc_list.append(int(trial_tc))

            if not trial_eps_list:
                continue

            B = len(trial_eps_list)
            eps_matrix = np.tile(np.array(current_eps, dtype=np.float64), (B, 1))
            eps_matrix[:, j] = np.array(trial_eps_list, dtype=np.float64)

            states = simulate_batch(ops, eps_matrix)
            trial_fids = fidelities_from_states(states)

            chosen = None
            for k in range(B):
                fid = float(trial_fids[k])
                if fid >= TARGET_FIDELITY + FIDELITY_MARGIN:
                    chosen = k
                    break
                if fid >= TARGET_FIDELITY - FIDELITY_MARGIN:
                    trial_eps_vec = current_eps.copy()
                    trial_eps_vec[j] = trial_eps_list[k]
                    qc_tmp = build_circuit(ops, trial_eps_vec)
                    if fidelity_exact_qiskit(qc_tmp) >= TARGET_FIDELITY:
                        chosen = k
                        break

            if chosen is not None:
                new_eps = trial_eps_list[chosen]
                new_tc  = trial_tc_list[chosen]
                current_eps[j] = new_eps
                current_t += new_tc - orig_tc
                improved = True

        if not improved:
            break

    qc_final = build_circuit(ops, current_eps)
    fid_final = fidelity_exact_qiskit(qc_final)

    return (cid, (current_t, fid_final, ops, rotation_indices, current_eps))

def worker_task(args):
    cid, cand_matrix = args
    ops = extract_ops(cand_matrix)
    return optimize_candidate_ops(ops, cid)

def _default_mp_start():
    methods = mp.get_all_start_methods()
    return "fork" if "fork" in methods else "spawn"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=_os.cpu_count() or 1,
                        help="Number of processes. Default: os.cpu_count().")
    parser.add_argument("--mp-start", type=str, default=_default_mp_start(),
                        choices=mp.get_all_start_methods(),
                        help="Multiprocessing start method.")
    parser.add_argument("--quiet", action="store_true", help="Less printing.")
    args = parser.parse_args()

    candidates = generate_candidates(_target_sv, N_CANDIDATES, CANDIDATE_SEED)

    ctx = mp.get_context(args.mp_start)
    tasks = [(i, candidates[i]) for i in range(N_CANDIDATES)]

    print(f"CPU workers:      {args.workers} (mp-start={args.mp_start})")
    print(f"Statevector:      {np.round(_target_sv, 6)}")
    print(f"Candidates:       {N_CANDIDATES}  |  fidelity threshold: {TARGET_FIDELITY}")
    print("=" * 60)

    results = {}

    t0 = time.perf_counter()
    with ctx.Pool(processes=args.workers) as pool:
        for cid, res in pool.imap_unordered(worker_task, tasks, chunksize=1):
            results[cid] = res

            if args.quiet:
                continue

            if res is None:
                print(f"  [{cid:2d}] SKIP – fidelity threshold not reached")
            else:
                tc, fid, ops, rot_idx, eps_list = res
                n_rot = len(rot_idx)
                n_cx = sum(1 for op in ops if op[0] == "cx")
                print(f"  [{cid:2d}] T={tc:4d}  fidelity={fid:.10f}  ({n_rot} rot, {n_cx} cx)")
    t1 = time.perf_counter()

    for cid in range(N_CANDIDATES):
        res = results.get(cid)
        if res is None:
            continue
        tc, fid, ops, rot_idx, eps_list = res
        if best is None:
            best = (tc, fid, cid, ops, rot_idx, eps_list)
            continue
        if tc < best[0] or (tc == best[0] and (fid > best[1] or (fid == best[1] and cid < best[2]))):
            best = (tc, fid, cid, ops, rot_idx, eps_list)

    if best is None:
        print("ERROR: no candidate met the fidelity target.")
        return

    tc, fid, cid, ops, rot_idx, eps_list = best
    qc = build_circuit(ops, eps_list)

    qasm3_str  = dumps3(qc)
    verified_t = count_t_gates_manual(qasm3_str)

    print(f"\n{'=' * 52}")
    print(f"  FINAL RESULT  (candidate {cid})")
    print(f"{'=' * 52}")
    print(f"  T-count  (sum):         {tc}")
    print(f"  T-count  (QASM):        {verified_t}")
    print(f"  Fidelity |⟨ψ|U|00⟩|²:  {fid:.10f}")
    print(f"  Runtime:               {t1 - t0:.3f}s")
    print(f"  Per-rotation breakdown:")
    for j, idx in enumerate(rot_idx):
        axis  = ops[idx][0]
        angle = ops[idx][2]
        _, tc_j = synthesize(axis, angle, eps_list[j])
        print(f"    rot[{j}] {axis}({angle:+.6f}) q{ops[idx][1]}: eps={eps_list[j]:.1e}  T={tc_j}")

    _os.makedirs("qasm", exist_ok=True)
    with open("qasm/unitary7.qasm", "w") as f:
        f.write(qasm3_str)
    print("\n  Saved to qasm/unitary7.qasm")


if __name__ == "__main__":
    main()
