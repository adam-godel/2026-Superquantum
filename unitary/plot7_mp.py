"""
plot7_mp.py -- multiprocessing plot for unitary 7 (state preparation)

Generates a T-count vs infidelity (1 - fidelity) plot using the same
visual style as plot10.py.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from plot10 import apply_scientific_style
from unitary7 import (
    generate_candidates,
    extract_ops,
    optimize_candidate_ops,
    _target_sv,
    N_CANDIDATES,
    CANDIDATE_SEED,
    TARGET_FIDELITY,
)


def _worker_task(args):
    cid, cand_matrix = args
    ops = extract_ops(cand_matrix)
    return optimize_candidate_ops(ops, cid)


def _collect_points(candidates, workers, mp_start):
    ctx = mp.get_context(mp_start)
    tasks = [(i, candidates[i]) for i in range(len(candidates))]
    results = []

    if workers == 1:
        for task in tasks:
            results.append(_worker_task(task))
    else:
        with ctx.Pool(processes=workers) as pool:
            for res in pool.imap_unordered(_worker_task, tasks, chunksize=1):
                results.append(res)

    points = []
    for cid, res in results:
        if res is None:
            continue
        tc, fid, _, _, _ = res
        points.append((int(tc), float(fid)))

    return points


def plot_unitary7(out_path, n_candidates, seed, workers, mp_start):
    candidates = generate_candidates(_target_sv, n_candidates, seed)
    points = _collect_points(candidates, workers, mp_start)

    if not points:
        print("No candidates met the fidelity target; no plot generated.")
        return

    best_points = {}
    for t, fid in points:
        if t not in best_points or fid > best_points[t]:
            best_points[t] = fid

    sorted_t = sorted(best_points.keys())
    sorted_fid = [best_points[t] for t in sorted_t]

    apply_scientific_style()
    plt.figure(figsize=(6.8, 4.8))

    plt.plot(
        sorted_t,
        sorted_fid,
        marker="o",
        linewidth=1.6,
        markersize=5,
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        label="Unitary 7",
    )

    plt.xscale("log")
    plt.xlabel("T-Count (Gates)", fontsize=12, fontweight="bold")
    plt.ylabel("Fidelity", fontsize=12, fontweight="bold")
    plt.title(
        f"Unitary 7: Fidelity vs T-Count\n(target fidelity >= {TARGET_FIDELITY:.6f})",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, which="both")
    plt.legend(loc="best", fontsize=11)
    fid_min = float(np.min(sorted_fid))
    fid_max = float(np.max(sorted_fid))
    if fid_max == fid_min:
        pad = max(1e-12, fid_min * 1e-9)
    else:
        pad = 0.10 * (fid_max - fid_min)
    fid_min = max(0.0, fid_min - pad)
    fid_max = min(1.0, fid_max + pad)
    plt.ylim(fid_min, fid_max)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="y", style="plain")

    if sorted_t:
        min_t_idx = 0
        max_t_idx = len(sorted_t) - 1
        textstr = f"Min T: {sorted_t[min_t_idx]} (fid={sorted_fid[min_t_idx]:.9f})\n"
        textstr += f"Max T: {sorted_t[max_t_idx]} (fid={sorted_fid[max_t_idx]:.9f})"
        plt.text(
            0.08,
            0.05,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
        )

        # Annotate three analytical points: min T, max T, best fidelity
        max_fid_idx = int(np.argmax(sorted_fid))
        label_indices = []
        for idx in (min_t_idx, max_t_idx, max_fid_idx):
            if idx not in label_indices:
                label_indices.append(idx)
            if len(label_indices) == 3:
                break
        offsets = [(-18, 12), (-18, -12), (-26, 16)]
        for k, idx in enumerate(label_indices):
            t_val = sorted_t[idx]
            fid_val = sorted_fid[idx]
            dx, dy = offsets[k % len(offsets)]
            plt.annotate(
                f"{fid_val:.9f}",
                xy=(t_val, fid_val),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=int,
        default=N_CANDIDATES,
        help=f"Number of candidates (default: {N_CANDIDATES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CANDIDATE_SEED,
        help=f"Candidate RNG seed (default: {CANDIDATE_SEED}).",
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
    parser.add_argument(
        "--out",
        type=str,
        default="unitary_7_distance_vs_tcount.png",
        help="Output image file.",
    )
    args = parser.parse_args()

    plot_unitary7(
        out_path=args.out,
        n_candidates=args.candidates,
        seed=args.seed,
        workers=args.workers,
        mp_start=args.mp_start,
    )


if __name__ == "__main__":
    main()
