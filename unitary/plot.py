import numpy as np
import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.qasm3 import dumps as dumps3

from utils import Ry, Rz
from test import count_t_gates_manual, distance_global_phase, expected as EXPECTED_DICT

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
        # Unitary 7: Add construction here when you have it
        # This appears to be a custom unitary from a state vector
        pass
    
    elif uid == 10:
        # Unitary 10: Not defined in expected dict
        pass
        
    return qc


def run_plot(unitary_ids, theta, show_individual=True, show_combined=True):
    """
    Creates comprehensive plots showing distance vs T-count tradeoffs.
    
    Args:
        unitary_ids: List of unitary IDs to analyze
        theta: Angle parameter for constructions
        show_individual: If True, create individual plots for each construction
        show_combined: If True, create a combined comparison plot
    """
    # Wide epsilon range to explore the full tradeoff curve
    epsilons = [10**(-i/2) for i in range(2, 18)]
    
    # Try both optimization levels
    optimization_levels = [0, 1]
    
    all_results = {}
    
    for uid in unitary_ids:
        if uid not in EXPECTED_DICT:
            print(f"Warning: Unitary {uid} not in EXPECTED_DICT, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing Unitary {uid} (theta={theta:.4f} rad = {theta*180/math.pi:.2f}°)")
        print(f"{'='*60}")
        
        target_u = EXPECTED_DICT[uid]
        results_by_opt = {}
        
        for opt_level in optimization_levels:
            results = []
            
            for eps in epsilons:
                qc = get_circuit_construction(uid, theta, eps, optimization_level=opt_level)
                
                # Skip if construction not implemented
                if qc.num_qubits == 0:
                    continue
                
                qasm_str = dumps3(qc)
                t_count = count_t_gates_manual(qasm_str)
                
                # Distance calculation with global phase alignment
                actual = Operator(qc).data
                aligned = distance_global_phase(actual, target_u)
                dist = np.linalg.norm(aligned - target_u)
                
                results.append((eps, t_count, dist))
                
            if results:
                results_by_opt[opt_level] = results
                
                # Print summary for this optimization level
                opt_name = "Synthesized only" if opt_level == 0 else "With exact gates"
                print(f"\n{opt_name}:")
                print(f"  Epsilon range: {min(r[0] for r in results):.2e} to {max(r[0] for r in results):.2e}")
                print(f"  T-count range: {min(r[1] for r in results)} to {max(r[1] for r in results)}")
                print(f"  Distance range: {min(r[2] for r in results):.2e} to {max(r[2] for r in results):.2e}")
        
        all_results[uid] = results_by_opt
        
        # Individual plot for this construction
        if show_individual and results_by_opt:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            for opt_level, results in results_by_opt.items():
                # Filter: Keep best distance for each unique T-count
                best_points = {}
                for eps, t, d in results:
                    # Convert distance to scalar if it's an array
                    d_val = float(d) if hasattr(d, '__len__') else d
                    if t not in best_points or d_val < best_points[t][0]:
                        best_points[t] = (d_val, eps)
                
                sorted_t = sorted(best_points.keys())
                sorted_d = [best_points[t][0] for t in sorted_t]
                sorted_eps = [best_points[t][1] for t in sorted_t]
                
                opt_label = "Synthesized" if opt_level == 0 else "With exact gates"
                
                # Plot 1: Distance vs T-count (main plot)
                ax1.plot(sorted_t, sorted_d, marker='o', linewidth=2, 
                        markersize=6, label=opt_label, alpha=0.8)
                
                # Plot 2: Both T-count and distance vs epsilon (to show relationship)
                ax2_twin = ax2.twinx()
                line1 = ax2.plot(sorted_eps, sorted_t, marker='s', linewidth=2,
                               label=f'{opt_label} (T-count)', alpha=0.7)
                line2 = ax2_twin.plot(sorted_eps, sorted_d, marker='^', linewidth=2,
                                     linestyle='--', label=f'{opt_label} (Distance)', alpha=0.7)
            
            # Format Plot 1
            ax1.set_yscale('log')
            ax1.set_xlabel('T-Count (Gates)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Distance to Target (Error)', fontsize=12, fontweight='bold')
            ax1.set_title(f'Unitary {uid}: Distance vs T-Count Tradeoff', fontsize=13, fontweight='bold')
            ax1.grid(True, which="both", ls="--", alpha=0.3)
            ax1.legend(loc='best')
            
            # Format Plot 2
            ax2.set_xscale('log')
            ax2.set_xlabel('Epsilon (Synthesis Precision)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('T-Count', fontsize=12, fontweight='bold', color='tab:blue')
            ax2_twin.set_ylabel('Distance', fontsize=12, fontweight='bold', color='tab:orange')
            ax2_twin.set_yscale('log')
            ax2.set_title(f'Unitary {uid}: Effect of Epsilon', fontsize=13, fontweight='bold')
            ax2.grid(True, which="both", ls="--", alpha=0.3)
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            ax2_twin.tick_params(axis='y', labelcolor='tab:orange')
            
            plt.tight_layout()
            plt.savefig(f'unitary_{uid}_analysis.png', dpi=150, bbox_inches='tight')
            print(f"  Saved plot: unitary_{uid}_analysis.png")
    
    # Combined comparison plot
    if show_combined and all_results:
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        
        for idx, (uid, results_by_opt) in enumerate(all_results.items()):
            # Use optimization level 1 (with exact gates) for the combined view
            opt_level = 1 if 1 in results_by_opt else 0
            results = results_by_opt[opt_level]
            
            # Filter: Keep best distance for each unique T-count
            best_points = {}
            for eps, t, d in results:
                # Convert distance to scalar if it's an array
                d_val = float(d) if hasattr(d, '__len__') else d
                if t not in best_points or d_val < best_points[t]:
                    best_points[t] = d_val
            
            sorted_t = sorted(best_points.keys())
            sorted_d = [best_points[t] for t in sorted_t]
            
            plt.plot(sorted_t, sorted_d, marker='o', linewidth=2.5, 
                    markersize=8, label=f'Unitary {uid}', 
                    color=colors[idx], alpha=0.8)
        
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('T-Count (Gates)', fontsize=13, fontweight='bold')
        plt.ylabel('Distance to Target (Error)', fontsize=13, fontweight='bold')
        plt.title(f'All Constructions: Efficiency Frontier\n(θ = {theta:.4f} rad = {theta*180/math.pi:.2f}°)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(loc='best', fontsize=11, framealpha=0.9)
        plt.tight_layout()
        plt.savefig('all_constructions_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n{'='*60}")
        print("Saved combined plot: all_constructions_comparison.png")
        print(f"{'='*60}")
        
        plt.show()

if __name__ == "__main__":
    # List all constructions you want to investigate
    # Unitaries 5, 8, 9 don't use theta parameter (fixed constructions)
    # Unitary 7 needs custom implementation
    # Unitary 10 doesn't exist in expected dict
    constructions_to_analyze = [2, 3, 4, 5, 6, 8, 9]
    
    theta_value = math.pi / 7
    
    print(f"Running comprehensive analysis for θ = {theta_value:.4f} rad = {theta_value*180/math.pi:.2f}°")
    print(f"Constructions: {constructions_to_analyze}")
    print(f"Note: Unitaries 5, 8, 9 are fixed constructions (don't depend on theta or epsilon)")
    print(f"\nThis will generate:")
    print(f"  - Individual analysis plots for each construction")
    print(f"  - Combined comparison plot showing all constructions")
    print(f"  - Exploration of different epsilon values and optimization strategies")
    
    run_plot(constructions_to_analyze, theta_value, 
             show_individual=True, show_combined=True)