import numpy as np
import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.qasm3 import dumps as dumps3

from utils import Ry, Rz
from test import count_t_gates_manual, distance_global_phase, expected as EXPECTED_DICT


def optimize_rz_gate(angle, epsilon):
    """
    Determines the best T/S gate decomposition for a given Rz angle.
    Returns: (circuit_fragment, t_count, description)
    """
    norm_angle = angle % (2 * math.pi)
    
    # Check for exact special cases (zero T-cost)
    if np.isclose(norm_angle, 0, atol=1e-10):
        return None, 0, "identity"
    if np.isclose(norm_angle, math.pi/2, atol=1e-10):
        return lambda qc, q: qc.s(q), 0, "S"
    if np.isclose(norm_angle, 3*math.pi/2, atol=1e-10):
        return lambda qc, q: qc.sdg(q), 0, "S†"
    if np.isclose(norm_angle, math.pi/4, atol=1e-10):
        return lambda qc, q: qc.t(q), 1, "T"
    if np.isclose(norm_angle, 7*math.pi/4, atol=1e-10):
        return lambda qc, q: qc.tdg(q), 1, "T†"
    
    # For other angles, use Rz synthesis
    rz_circuit = Rz(angle, epsilon)
    qasm_str = dumps3(rz_circuit)
    t_count = count_t_gates_manual(qasm_str)
    
    return lambda qc, q: qc.append(rz_circuit.to_gate(), [q]), t_count, f"Rz(eps={epsilon:.2e})"


def get_circuit_with_optimization(unitary_id, theta, eps):
    """
    Constructs circuits with optimized T/S gate replacements.
    Returns: (circuit, total_t_count, optimization_details)
    """
    qc = QuantumCircuit(2)
    total_t_count = 0
    optimizations = []
    
    def apply_optimized_rz(circuit, angle, epsilon, qubit, label=""):
        nonlocal total_t_count
        gate_func, t_cost, desc = optimize_rz_gate(angle, epsilon)
        if gate_func:
            gate_func(circuit, qubit)
        total_t_count += t_cost
        if label:
            optimizations.append(f"{label}: {desc} (T={t_cost})")
    
    # Construction implementations with Rz tracking
    if unitary_id == 2:
        # Ry gates (these use Rz internally)
        ry1 = Ry(theta/2, eps)
        ry2 = Ry(-theta/2, eps)
        qc.append(ry1.to_gate(), [0])
        qc.cx(1, 0)
        qc.append(ry2.to_gate(), [0])
        qc.cx(1, 0)
        
        # Count T-gates from Ry decompositions
        qasm_str = dumps3(qc)
        total_t_count = count_t_gates_manual(qasm_str)

    elif unitary_id == 3:
        qc.cx(0, 1)
        apply_optimized_rz(qc, -2 * theta, eps, 1, "Rz(-2θ)")
        qc.cx(0, 1)

    elif unitary_id == 4:
        # Two layers of H-CX-Rz-CX-H
        for layer in range(2):
            qc.h([0, 1])
            qc.cx(0, 1)
            apply_optimized_rz(qc, -2 * theta, eps, 1, f"Layer{layer+1}")
            qc.cx(0, 1)
            qc.h([0, 1])

    elif unitary_id == 6:
        qc.h([0, 1])
        qc.cx(0, 1)
        apply_optimized_rz(qc, -2 * theta, eps, 1, "Rz(-2θ)")
        qc.cx(0, 1)
        qc.h([0, 1])
        apply_optimized_rz(qc, -theta, eps, 0, "Rz(-θ) q0")
        apply_optimized_rz(qc, -theta, eps, 1, "Rz(-θ) q1")

    elif unitary_id == 7:
        # Example construction 7 (you can customize)
        qc.h([0, 1])
        apply_optimized_rz(qc, -theta, eps, 0, "Rz(-θ) q0")
        qc.cx(0, 1)
        apply_optimized_rz(qc, -theta, eps, 1, "Rz(-θ) q1")
        qc.cx(0, 1)
        qc.h([0, 1])

    elif unitary_id == 10:
        # Example construction 10 (you can customize)
        for i in range(3):
            qc.cx(0, 1)
            apply_optimized_rz(qc, -theta, eps, 1, f"Rz(-θ) iter{i}")
            qc.cx(0, 1)
    
    return qc, total_t_count, optimizations


def run_comprehensive_analysis(unitary_ids, theta, epsilon_range=None):
    """
    Runs comprehensive analysis across all constructions and epsilon values.
    """
    if epsilon_range is None:
        # More granular epsilon sampling for better curves
        epsilon_range = [10**(-i/2) for i in range(2, 20)]
    
    plt.figure(figsize=(12, 8))
    
    # Track best configuration for each construction
    results_summary = {}
    
    for uid in unitary_ids:
        if uid not in EXPECTED_DICT:
            print(f"Warning: Unitary {uid} not in EXPECTED_DICT, skipping...")
            continue
        
        t_counts = []
        distances = []
        epsilon_used = []
        target_u = EXPECTED_DICT[uid]
        
        print(f"\n{'='*60}")
        print(f"Analyzing Unitary {uid}")
        print(f"{'='*60}")

        for eps in epsilon_range:
            qc, t_count, opts = get_circuit_with_optimization(uid, theta, eps)
            
            # Calculate distance
            actual = Operator(qc).data
            aligned = distance_global_phase(actual, target_u)
            dist = np.linalg.norm(aligned - target_u)
            
            t_counts.append(t_count)
            distances.append(dist)
            epsilon_used.append(eps)
            
            print(f"  ε={eps:.2e}: T-count={t_count:3d}, Distance={dist:.2e}")
        
        # Remove duplicates (same T-count) - keep the one with best distance
        unique_data = {}
        for t, d, e in zip(t_counts, distances, epsilon_used):
            if t not in unique_data or d < unique_data[t][0]:
                unique_data[t] = (d, e)
        
        # Sort by T-count for proper line plotting
        sorted_t = sorted(unique_data.keys())
        plot_distances = [unique_data[t][0] for t in sorted_t]
        
        # Store results
        results_summary[uid] = {
            't_counts': sorted_t,
            'distances': plot_distances,
            'min_dist': min(plot_distances),
            'max_t': max(sorted_t)
        }
        
        # Plot
        plt.plot(sorted_t, plot_distances, marker='o', linewidth=2, 
                markersize=6, label=f'Construction {uid}', alpha=0.8)
        
        print(f"  Best: T={min(sorted_t)} at distance {max(plot_distances):.2e}")
        print(f"  Worst: T={max(sorted_t)} at distance {min(plot_distances):.2e}")
    
    # Formatting
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('T-Count (Gates)', fontsize=12, fontweight='bold')
    plt.ylabel('Distance to Target (Error)', fontsize=12, fontweight='bold')
    plt.title(f'Construction Comparison: Distance vs T-Count\n(θ = {theta:.4f} rad = {theta*180/math.pi:.2f}°)', 
            fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for uid, data in results_summary.items():
        print(f"Construction {uid}:")
        print(f"  T-count range: {min(data['t_counts'])} - {data['max_t']}")
        print(f"  Best distance: {data['min_dist']:.2e}")
        print(f"  Efficiency: {data['min_dist']/data['max_t']:.2e} (dist/T)")


def compare_specific_epsilon(unitary_ids, theta, epsilon_values):
    """
    Compares constructions at specific epsilon values.
    """
    fig, axes = plt.subplots(1, len(epsilon_values), figsize=(6*len(epsilon_values), 5))
    if len(epsilon_values) == 1:
        axes = [axes]
    
    for idx, eps in enumerate(epsilon_values):
        ax = axes[idx]
        
        t_counts = []
        distances = []
        labels = []
        
        for uid in unitary_ids:
            if uid not in EXPECTED_DICT:
                continue
            
            qc, t_count, _ = get_circuit_with_optimization(uid, theta, eps)
            
            actual = Operator(qc).data
            target_u = EXPECTED_DICT[uid]
            aligned = distance_global_phase(actual, target_u)
            dist = np.linalg.norm(aligned - target_u)
            
            t_counts.append(t_count)
            distances.append(dist)
            labels.append(f'U{uid}')
        
        ax.scatter(t_counts, distances, s=100, alpha=0.7)
        for i, label in enumerate(labels):
            ax.annotate(label, (t_counts[i], distances[i]), 
                    xytext=(5, 5), textcoords='offset points')
        
        ax.set_yscale('log')
        ax.set_xlabel('T-Count')
        ax.set_ylabel('Distance')
        ax.set_title(f'ε = {eps:.2e}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Theta value (you can change this)
    theta_val = math.pi / 7
    
    # All constructions to investigate
    constructions = [2, 3, 4, 6, 7, 10]
    
    print(f"Running comprehensive analysis for θ = {theta_val:.4f} rad")
    print(f"Constructions: {constructions}")
    
    # Main analysis
    run_comprehensive_analysis(constructions, theta_val)
    
    # Optional: Compare at specific epsilon values
    print("\nGenerating epsilon-specific comparison...")
    compare_specific_epsilon(constructions, theta_val, [1e-2, 1e-4, 1e-6])