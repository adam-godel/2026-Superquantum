"""
plot10.py -- plot T-count vs distance for unitary10 optimization

This mirrors the plotting structure in plot.py but uses the unitary10
optimization logic (uniform sweep + greedy per-rotation relaxation) and
plots the final (T, distance) point for each target distance.
"""

# --- perf hygiene: avoid each process spawning many BLAS threads -----------
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import os
import argparse
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, quantum_info, transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate

from optim import _synthesize, normalize_angle, build_circuit, total_t_count

# --- Configuration (same values as unitary10.py) ---
ANGLE_TOL = 1e-9

TARGET_DISTANCES = [
    1e-1, 7e-2, 5e-2, 3e-2, 2e-2, 1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3
]

# Ultra-granular epsilon sweep: 100 points from 1e-1 to 1e-9
EPS_COARSE_SWEEP = np.logspace(-1, -9, 100)

RELAXATION_FACTORS = [
    100, 50, 30, 20, 15, 10, 7, 5, 3, 2, 1.5, 1.3, 1.2, 1.1, 1.05, 1.02
]


def apply_scientific_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.top": True,
        "ytick.right": True,
        "grid.color": "0.85",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
    })


def operator_distance(actual, reference):
    d = actual.shape[0]
    inner = np.trace(actual.conj().T @ reference)
    return float(np.sqrt(max(2 * d - 2 * np.abs(inner), 0.0)))


def extract_ops(target_matrix):
    template_qc = QuantumCircuit(2)
    template_qc.append(UnitaryGate(target_matrix), [0, 1])
    template = transpile(
        template_qc,
        basis_gates=["u3", "cx"],
        optimization_level=2,
        seed_transpiler=0,
    )

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
            a = normalize_angle(float(inst.operation.params[0]))
            if abs(a) > ANGLE_TOL:
                ops.append(("rz", qubits[0], a))
        # unknown/ignored ops are skipped to match unitary10.py behavior

    return ops


def optimize_for_target(target_dist, ops, rotation_indices, target):
    n_rotations = len(rotation_indices)
    cache = {}

    # Phase 1: Uniform sweep
    best_uniform_eps = None
    for eps in EPS_COARSE_SWEEP:
        eps_list = [float(eps)] * n_rotations
        qc = build_circuit(ops, eps_list, cache)
        dist = operator_distance(Operator(qc).data, target)
        if dist < target_dist:
            best_uniform_eps = float(eps)
            break

    if best_uniform_eps is None:
        return False, None, None

    # Phase 2: Per-rotation relaxation
    current_eps = [best_uniform_eps] * n_rotations
    current_t = total_t_count(ops, rotation_indices, current_eps, cache)
    current_dist = operator_distance(Operator(build_circuit(ops, current_eps, cache)).data, target)

    while True:
        any_improved = False
        for j in range(n_rotations):
            idx = rotation_indices[j]
            axis, angle, orig_eps = ops[idx][0], ops[idx][2], current_eps[j]
            _, orig_tc_j = _synthesize(axis, angle, orig_eps, cache)

            for factor in RELAXATION_FACTORS:
                trial_eps = orig_eps * factor
                if trial_eps > 0.5:
                    continue

                _, trial_tc_j = _synthesize(axis, angle, trial_eps, cache)
                if trial_tc_j >= orig_tc_j:
                    continue

                trial_eps_list = current_eps.copy()
                trial_eps_list[j] = trial_eps
                trial_dist = operator_distance(
                    Operator(build_circuit(ops, trial_eps_list, cache)).data, target
                )

                if trial_dist < target_dist:
                    current_t = current_t - orig_tc_j + trial_tc_j
                    current_eps[j] = trial_eps
                    current_dist = trial_dist
                    any_improved = True
                    break
        if not any_improved:
            break

    final_qc = build_circuit(ops, current_eps, cache)
    final_t = total_t_count(ops, rotation_indices, current_eps, cache)
    final_dist = operator_distance(Operator(final_qc).data, target)

    return True, final_t, final_dist


def _worker_task(args):
    target_dist, ops, rotation_indices, target = args
    success, final_t, final_dist = optimize_for_target(
        target_dist, ops, rotation_indices, target
    )
    return (target_dist, success, final_t, final_dist)


def run_plot(target_distances, show_plot=True, workers=1, mp_start="spawn"):
    unitary = quantum_info.random_unitary(4, seed=42)
    target = unitary.data

    ops = extract_ops(target)
    rotation_indices = [i for i, op in enumerate(ops) if op[0] in ("rx", "ry", "rz")]

    print(f"Rotations to optimize: {len(rotation_indices)}")

    results = []
    tasks = [(td, ops, rotation_indices, target) for td in target_distances]

    if workers is None:
        workers = os.cpu_count() or 1

    if workers > 1:
        ctx = mp.get_context(mp_start)
        with ctx.Pool(processes=workers) as pool:
            raw_results = list(pool.imap_unordered(_worker_task, tasks, chunksize=1))
    else:
        raw_results = [_worker_task(task) for task in tasks]

    for target_dist, success, final_t, final_dist in sorted(raw_results, key=lambda r: r[0]):
        print("\n" + "=" * 60)
        print(f"RUNNING OPTIMIZATION FOR TARGET DISTANCE: {target_dist}")
        print("=" * 60)
        if not success:
            print(f"Warning: Target distance {target_dist} not reachable in sweep.")
            continue

        print(f"DONE -> Target: {target_dist} | Final T: {final_t} | Final Dist: {final_dist:.6e}")
        results.append((target_dist, final_t, final_dist))

    if not results:
        print("\nNo results to plot.")
        return

    if show_plot:
        apply_scientific_style()
        # Plot T-count vs distance for each target distance
        results_sorted = sorted(results, key=lambda r: r[1])
        t_vals = [r[1] for r in results_sorted]
        d_vals = [r[2] for r in results_sorted]
        labels = [r[0] for r in results_sorted]

        plt.figure(figsize=(6.8, 4.8))
        plt.plot(
            t_vals,
            d_vals,
            marker="o",
            linewidth=1.6,
            markersize=5,
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
        )
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("T-Count (Gates)", fontsize=12, fontweight="bold")
        plt.ylabel("Distance to Target (Error)", fontsize=12, fontweight="bold")
        plt.title("Unitary 10: Distance vs T-Count", fontsize=13, fontweight="bold")
        plt.grid(True, which="both")

        # Annotate points with their target distance
        for t, d, td in zip(t_vals, d_vals, labels):
            plt.annotate(
                f"{td:.1e}",
                xy=(t, d),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="0.6", alpha=0.9),
            )

        plt.tight_layout()
        filename = "unitary10_distance_vs_tcount.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"\nSaved plot: {filename}")
        plt.close()

    print("\nAll target optimizations complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes (default: os.cpu_count()).",
    )
    parser.add_argument(
        "--mp-start",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method (default: spawn).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the plot and only print results.",
    )
    args = parser.parse_args()

    run_plot(
        TARGET_DISTANCES,
        show_plot=not args.no_plot,
        workers=args.workers,
        mp_start=args.mp_start,
    )
