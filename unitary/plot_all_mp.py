"""
plot_all_mp.py -- multiprocessing plot generation for multiple unitaries

Uses the same visual style as plot10.py (scientific look, black line with
white markers) and writes one image per unitary.
"""

# --- perf hygiene: avoid each process spawning many BLAS threads -----------
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import os
import math
import argparse
import multiprocessing as mp
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit import quantum_info
from qiskit.quantum_info import Operator
from qiskit.qasm3 import dumps as dumps3

from utils import Ry, Rz
from test import count_t_gates_manual, distance_global_phase, expected as EXPECTED_DICT
from plot10 import apply_scientific_style

ANGLE_TOL = 1e-9
EPSILONS = [10**(-i/2) for i in range(2, 20)]

FIXED_UNITARIES = {5, 8, 9}
SUPPORTED_UNITARIES = {2, 3, 4, 5, 6, 8, 9}


def smart_rz(qc, angle, eps, qubit, use_exact=True):
    norm_angle = angle % (2 * math.pi)
    if use_exact:
        if np.isclose(norm_angle, 0, atol=1e-10):
            return
        if np.isclose(norm_angle, math.pi/2, atol=1e-10):
            qc.s(qubit)
            return
        if np.isclose(norm_angle, math.pi, atol=1e-10):
            qc.z(qubit)
            return
        if np.isclose(norm_angle, 3*math.pi/2, atol=1e-10):
            qc.sdg(qubit)
            return
        if np.isclose(norm_angle, math.pi/4, atol=1e-10):
            qc.t(qubit)
            return
        if np.isclose(norm_angle, 7*math.pi/4, atol=1e-10):
            qc.tdg(qubit)
            return
    qc.append(Rz(angle, eps).to_gate(), [qubit])


def smart_ry(qc, angle, eps, qubit, use_exact=True):
    norm_angle = angle % (2 * math.pi)
    if use_exact:
        if np.isclose(norm_angle, 0, atol=1e-10):
            return
        if np.isclose(norm_angle, math.pi, atol=1e-10):
            qc.y(qubit)
            return
    qc.append(Ry(angle, eps).to_gate(), [qubit])


def get_circuit_construction(uid, theta, eps, optimization_level=1):
    qc = QuantumCircuit(2)
    use_exact = (optimization_level >= 1)

    if uid == 2:
        smart_ry(qc, theta/2, eps, 1, use_exact)
        qc.cx(0, 1)
        smart_ry(qc, -theta/2, eps, 1, use_exact)
        qc.cx(0, 1)
    elif uid == 3:
        qc.cx(0, 1)
        smart_rz(qc, -2*theta, eps, 1, use_exact)
        qc.cx(0, 1)
    elif uid == 4:
        qc.h(0); qc.h(1)
        qc.s(0); qc.s(1)
        qc.h(0); qc.h(1)
        qc.cx(0, 1)
        smart_rz(qc, -2*theta, eps, 1, use_exact)
        qc.cx(0, 1)
        qc.h(0); qc.h(1)
        qc.sdg(0); qc.sdg(1)
        qc.h(0); qc.h(1)

        qc.h(0); qc.h(1)
        qc.cx(0, 1)
        smart_rz(qc, -2*theta, eps, 1, use_exact)
        qc.cx(0, 1)
        qc.h(0); qc.h(1)
    elif uid == 6:
        qc.h(0); qc.h(1)
        qc.cx(0, 1)
        smart_rz(qc, -2*theta, eps, 1, use_exact)
        qc.cx(0, 1)
        qc.h(0); qc.h(1)

        smart_rz(qc, -theta, eps, 0, use_exact)
        smart_rz(qc, -theta, eps, 1, use_exact)
    elif uid == 5:
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.cx(0, 1)
    elif uid == 8:
        qc.h(1)
        qc.t(0)
        qc.t(1)
        qc.cx(0, 1)
        qc.tdg(1)
        qc.cx(0, 1)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.cx(0, 1)
    elif uid == 9:
        qc.h(0)
        qc.t(0)
        qc.t(1)
        qc.cx(1, 0)
        qc.tdg(0)
        qc.cx(1, 0)
        qc.h(0)
        qc.s(0)
        qc.s(1)
        qc.t(1)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.cx(0, 1)
    else:
        return QuantumCircuit(0)

    return qc


def compute_results(uid, theta):
    if uid not in EXPECTED_DICT:
        return (uid, False, "not in expected dict", None)
    if uid not in SUPPORTED_UNITARIES:
        return (uid, False, "no construction for this unitary", None)

    target_u = EXPECTED_DICT[uid]
    results = []

    if uid in FIXED_UNITARIES:
        qc = get_circuit_construction(uid, theta, EPSILONS[0], optimization_level=1)
        if qc.num_qubits == 0:
            return (uid, False, "construction returned empty circuit", None)
        qasm_str = dumps3(qc)
        t_count = count_t_gates_manual(qasm_str)
        actual = Operator(qc).data
        aligned = distance_global_phase(actual, target_u)
        dist = np.linalg.norm(aligned - target_u)
        d_val = float(dist) if hasattr(dist, "__len__") else dist
        results.append((EPSILONS[0], t_count, d_val))
    else:
        for eps in EPSILONS:
            qc = get_circuit_construction(uid, theta, eps, optimization_level=1)
            if qc.num_qubits == 0:
                continue
            qasm_str = dumps3(qc)
            t_count = count_t_gates_manual(qasm_str)
            actual = Operator(qc).data
            aligned = distance_global_phase(actual, target_u)
            dist = np.linalg.norm(aligned - target_u)
            d_val = float(dist) if hasattr(dist, "__len__") else dist
            results.append((eps, t_count, d_val))

    if not results:
        return (uid, False, "no results generated", None)

    best_points = {}
    for _, t, d in results:
        if t not in best_points or d < best_points[t]:
            best_points[t] = d

    sorted_t = sorted(best_points.keys())
    sorted_d = [best_points[t] for t in sorted_t]
    return (uid, True, None, (sorted_t, sorted_d))


def plot_unitary(uid, theta, out_dir, run_tag):
    uid, ok, reason, payload = compute_results(uid, theta)
    if not ok:
        return (uid, False, reason)

    sorted_t, sorted_d = payload
    apply_scientific_style()
    plt.figure(figsize=(6.8, 4.8))

    plt.plot(
        sorted_t,
        sorted_d,
        marker="o",
        linewidth=1.6,
        markersize=5,
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        label=f"Unitary {uid}",
    )

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("T-Count (Gates)", fontsize=12, fontweight="bold")
    plt.ylabel("Distance to Target (Error)", fontsize=12, fontweight="bold")
    plt.title(
        f"Unitary {uid}: Distance vs T-Count\n(theta = {theta:.4f} rad = {theta*180/math.pi:.2f} deg)",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, which="both")
    plt.legend(loc="best", fontsize=11)

    if sorted_t:
        min_t_idx = 0
        max_t_idx = len(sorted_t) - 1
        textstr = f"Min T: {sorted_t[min_t_idx]} (dist={sorted_d[min_t_idx]:.2e})\n"
        textstr += f"Max T: {sorted_t[max_t_idx]} (dist={sorted_d[max_t_idx]:.2e})"
        plt.text(
            0.08,
            0.05,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
        )

    plt.tight_layout()
    filename = os.path.join(out_dir, f"unitary_{uid}_distance_vs_tcount_{run_tag}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    return (uid, True, filename)


def _worker_task(args):
    uid, theta, out_dir, run_tag = args
    return plot_unitary(uid, theta, out_dir, run_tag)


def _parse_unitary_list(text):
    if text.strip().lower() == "all":
        return sorted(EXPECTED_DICT.keys())
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unitaries",
        type=str,
        default="2,3,4,5,6,8,9",
        help="Comma-separated unitary IDs, or 'all'. Default: 2,3,4,5,6,8,9",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=math.pi / 7,
        help="Theta parameter for constructions (default: pi/7).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output directory for images (default: current directory).",
    )
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
    args = parser.parse_args()

    unitary_ids = _parse_unitary_list(args.unitaries)
    os.makedirs(args.out_dir, exist_ok=True)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    tasks = [(uid, args.theta, args.out_dir, run_tag) for uid in unitary_ids]
    ctx = mp.get_context(args.mp_start)

    with ctx.Pool(processes=args.workers) as pool:
        for uid, ok, info in pool.imap_unordered(_worker_task, tasks, chunksize=1):
            if ok:
                print(f"[unitary {uid}] saved: {info}")
            else:
                print(f"[unitary {uid}] skipped: {info}")


if __name__ == "__main__":
    main()
