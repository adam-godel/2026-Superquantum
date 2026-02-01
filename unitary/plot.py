import numpy as np
import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit import quantum_info
from qiskit.quantum_info import Operator
from qiskit.qasm3 import dumps as dumps3

from utils import Ry, Rz
from test import count_t_gates_manual, distance_global_phase, expected as EXPECTED_DICT
from unitary7 import generate_candidates, optimize_candidate_ops

from plot10 import apply_scientific_style

def smart_rz(qc, angle, eps, qubit, use_exact=True):
    """
    Intelligently choose between exact T/S gates and synthesized Rz.
    If use_exact=True and angle is a multiple of pi/4, use exact gates.
    Otherwise, synthesize with given epsilon.
    """
    norm_angle = angle % (2 * math.pi)
    
    if use_exact:
        # Check for exact gate possibilities (zero T-cost or minimal T-cost)
        if np.isclose(norm_angle, 0, atol=1e-10): 
            return  # Identity, do nothing
        elif np.isclose(norm_angle, math.pi/2, atol=1e-10):   
            qc.s(qubit)
            return
        elif np.isclose(norm_angle, math.pi, atol=1e-10):     
            qc.z(qubit)
            return
        elif np.isclose(norm_angle, 3*math.pi/2, atol=1e-10): 
            qc.sdg(qubit)
            return
        elif np.isclose(norm_angle, math.pi/4, atol=1e-10):   
            qc.t(qubit)
            return
        elif np.isclose(norm_angle, 7*math.pi/4, atol=1e-10): 
            qc.tdg(qubit)
            return
    
    # Fallback to synthesized Rz gate
    qc.append(Rz(angle, eps).to_gate(), [qubit])

def smart_ry(qc, angle, eps, qubit, use_exact=True):
    """
    Intelligently choose between exact gates and synthesized Ry.
    """
    norm_angle = angle % (2 * math.pi)
    
    if use_exact:
        if np.isclose(norm_angle, 0, atol=1e-10): 
            return  # Identity
        elif np.isclose(norm_angle, math.pi, atol=1e-10):
            qc.y(qubit)
            return
    
    # Fallback to synthesized Ry gate
    qc.append(Ry(angle, eps).to_gate(), [qubit])


def get_circuit_construction(uid, theta, eps, optimization_level=0):
    """
    Constructions from optim.py with configurable optimization.
    
    optimization_level:
    0 = Always use synthesized Rz/Ry gates
    1 = Use exact T/S gates when angles align with pi/4 multiples
    """
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
        # Unitary 5: Simple SWAP-like construction (no rotation needed)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.cx(0, 1)
    
    elif uid == 8:
        # Unitary 8: Fixed gate construction with T gates
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
        # Unitary 9: Fixed gate construction with T and S gates
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
    
    elif uid == 7:
        # ── target (must match test.py case 7) ─────────────────────────────────────
        statevector = quantum_info.random_statevector(4, seed=42).data

        # ── tuning knobs ───────────────────────────────────────────────────────────
        N_CANDIDATES       = 50
        CANDIDATE_SEED     = 42
        TARGET_FIDELITY    = 0.9999
        ANGLE_TOL          = 1e-9

        EPS_COARSE = [10**(-i/2) for i in range(2, 18)]

        RELAXATION_FACTORS = [100, 50, 30, 20, 15, 10, 7, 5, 3, 2, 1.5, 1.3, 1.2, 1.1, 1.05, 1.02]

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

    
    elif uid == 10:
        # Unitary 10: Not defined in expected dict
        pass
        
    elif uid == 11:
        pass

    return qc


def run_plot(unitary_ids, theta, show_individual=True, show_combined=False):
    """
    Creates simple T-count vs Distance plots for each construction.
    
    Args:
        unitary_ids: List of unitary IDs to analyze
        theta: Angle parameter for constructions
        show_individual: If True, create individual plots for each construction
        show_combined: If True, create a combined comparison plot (deprecated)
    """
    # Apply scientific plotting style
    apply_scientific_style()
    
    # Wide epsilon range to explore the full tradeoff curve
    epsilons = [10**(-i/2) for i in range(2, 20)]  # More granular range
    
    # Fixed constructions that don't depend on epsilon
    fixed_unitaries = {5, 8, 9, 11}
    
    all_results = {}
    
    for uid in unitary_ids:
        if uid not in EXPECTED_DICT:
            print(f"Warning: Unitary {uid} not in EXPECTED_DICT, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing Unitary {uid} (theta={theta:.4f} rad = {theta*180/math.pi:.2f}°)")
        print(f"{'='*60}")
        
        target_u = EXPECTED_DICT[uid]
        results = []
        
        # For fixed unitaries, only compute once
        if uid in fixed_unitaries:
            qc = get_circuit_construction(uid, theta, epsilons[0], optimization_level=1)
            
            if qc.num_qubits == 0:
                print(f"  Construction not implemented for Unitary {uid}")
                continue
            
            qasm_str = dumps3(qc)
            t_count = count_t_gates_manual(qasm_str)
            
            # Distance calculation with global phase alignment
            actual = Operator(qc).data
            aligned = distance_global_phase(actual, target_u)
            dist = np.linalg.norm(aligned - target_u)
            d_val = float(dist) if hasattr(dist, '__len__') else dist
            
            # Add single result
            results.append((epsilons[0], t_count, d_val))
            
            print(f"  Fixed construction: T-count={t_count}, Distance={d_val:.2e}")
            print(f"  Note: This is a fixed circuit (epsilon does not affect T-count or distance)")
        
        else:
            # For variable unitaries, scan all epsilon values
            for eps in epsilons:
                qc = get_circuit_construction(uid, theta, eps, optimization_level=1)
                
                # Skip if construction not implemented
                if qc.num_qubits == 0:
                    continue
                
                qasm_str = dumps3(qc)
                t_count = count_t_gates_manual(qasm_str)
                
                # Distance calculation with global phase alignment
                actual = Operator(qc).data
                aligned = distance_global_phase(actual, target_u)
                dist = np.linalg.norm(aligned - target_u)
                
                # Convert distance to scalar if it's an array
                d_val = float(dist) if hasattr(dist, '__len__') else dist
                
                results.append((eps, t_count, d_val))
            
            if results:
                # Print summary
                print(f"  Epsilon range: {min(r[0] for r in results):.2e} to {max(r[0] for r in results):.2e}")
                print(f"  T-count range: {min(r[1] for r in results)} to {max(r[1] for r in results)}")
                print(f"  Distance range: {min(r[2] for r in results):.2e} to {max(r[2] for r in results):.2e}")
        
        if not results:
            print(f"  No results generated for Unitary {uid}")
            continue
        
        all_results[uid] = results
        
        # Create individual plot for this construction
        if show_individual:
            # For fixed unitaries with single point, create a simple marker plot
            if uid in fixed_unitaries and len(results) == 1:
                eps, t_count, d_val = results[0]
                apply_scientific_style()
                plt.figure(figsize=(6.8, 4.8))
                
                plt.scatter([t_count], [d_val], s=90, marker='o',
                           facecolors='white', edgecolors='black',
                           linewidths=1.2, zorder=5, label=f'Unitary {uid} (Fixed)')
                
                plt.yscale('log')
                plt.xscale('log')
                plt.xlabel('T-Count (Gates)', fontsize=12, fontweight='bold')
                plt.ylabel('Distance to Target (Error)', fontsize=12, fontweight='bold')
                plt.title(f'Unitary {uid}: Fixed Construction\n(T-count={t_count}, Distance={d_val:.2e})', 
                         fontsize=13, fontweight='bold')
                plt.grid(True, which="both")
                plt.legend(loc='best', fontsize=11)
                
                # Add annotation
                plt.annotate(
                    f'T={t_count}\nDist={d_val:.2e}',
                    xy=(t_count, d_val),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.6', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='0.4', lw=0.8),
                )
                
                plt.tight_layout()
                filename = f'unitary_{uid}_distance_vs_tcount.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"  Saved plot: {filename}")
                plt.close()
            
            else:
                # For variable unitaries, filter and plot normally
                # Filter: Keep best distance for each unique T-count
                best_points = {}
                for eps, t, d in results:
                    if t not in best_points or d < best_points[t]:
                        best_points[t] = d
                
                sorted_t = sorted(best_points.keys())
                sorted_d = [best_points[t] for t in sorted_t]
                
                # Create simple plot
                apply_scientific_style()
                plt.figure(figsize=(6.8, 4.8))
                
                plt.plot(
                    sorted_t,
                    sorted_d,
                    marker='o',
                    linewidth=1.6,
                    markersize=5,
                    color='black',
                    markerfacecolor='white',
                    markeredgecolor='black',
                    label=f'Unitary {uid}',
                )
                
                plt.yscale('log')
                plt.xscale('log')
                plt.xlabel('T-Count (Gates)', fontsize=12, fontweight='bold')
                plt.ylabel('Distance to Target (Error)', fontsize=12, fontweight='bold')
                plt.title(f'Unitary {uid}: Distance vs T-Count\n(θ = {theta:.4f} rad = {theta*180/math.pi:.2f}°)', 
                         fontsize=13, fontweight='bold')
                plt.grid(True, which="both")
                plt.legend(loc='best', fontsize=11)

                # Annotate a small, non-overlapping subset of points (left-biased)
                ax = plt.gca()
                if sorted_t:
                    idx_min_d = int(np.argmin(sorted_d))
                    idx_max_d = int(np.argmax(sorted_d))
                    label_indices = sorted(set([0, len(sorted_t) - 1, idx_min_d, idx_max_d]))
                    offsets = [(-28, 12), (-28, -12), (-36, 18), (-36, -18)]
                    for k, idx in enumerate(label_indices):
                        t_val = sorted_t[idx]
                        d_val = sorted_d[idx]
                        dx, dy = offsets[k % len(offsets)]
                        va = 'bottom' if dy > 0 else 'top'
                        ax.annotate(
                            f'{d_val:.2e}',
                            xy=(t_val, d_val),
                            xytext=(dx, dy),
                            textcoords='offset points',
                            fontsize=9,
                            horizontalalignment='right',
                            verticalalignment=va,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', color='0.4', lw=0.8),
                        )
                
                # Add text with best and worst points
                if sorted_t:
                    min_t_idx = 0
                    max_t_idx = len(sorted_t) - 1
                    textstr = f'Min T: {sorted_t[min_t_idx]} (dist={sorted_d[min_t_idx]:.2e})\n'
                    textstr += f'Max T: {sorted_t[max_t_idx]} (dist={sorted_d[max_t_idx]:.2e})'
                    plt.text(0.5, 0.05, textstr, transform=plt.gca().transAxes,
                            fontsize=9, verticalalignment='bottom',
                            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.6', alpha=0.9))
                
                plt.tight_layout()
                filename = f'unitary_{uid}_distance_vs_tcount.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"  Saved plot: {filename}")
                plt.close()  # Close the figure to free memory
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Generated {len(all_results)} individual plots.")
    print(f"{'='*60}")

if __name__ == "__main__":
    # List all constructions you want to investigate
    # Unitaries 5, 8, 9 don't use theta parameter (fixed constructions)
    constructions_to_analyze = [6]
    
    theta_value = math.pi / 7
    
    print(f"Running analysis for θ = {theta_value:.4f} rad = {theta_value*180/math.pi:.2f}°")
    print(f"Constructions: {constructions_to_analyze}")
    print(f"Note: Unitaries 5, 8, 9 are fixed constructions (don't depend on theta or epsilon)")
    print(f"\nThis will generate individual T-count vs Distance plots for each construction.")
    
    run_plot(constructions_to_analyze, theta_value, show_individual=True)
