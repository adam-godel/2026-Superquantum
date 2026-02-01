import numpy as np
import os
from qiskit import QuantumCircuit, quantum_info, transpile
from qiskit.quantum_info import Operator
from qiskit.qasm3 import dumps as dumps3
from qiskit.circuit.library import UnitaryGate

from utils import Rz, Ry, Rx
from test import count_t_gates_manual

# --- Setup ---
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

ANGLE_TOL = 1e-9

def normalize_angle(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)

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

rotation_indices = [i for i, op in enumerate(ops) if op[0] in ("rx", "ry", "rz")]
n_rotations = len(rotation_indices)
_cache: dict[tuple, tuple] = {}

def _synthesize(axis, angle, eps):
    key = (axis, angle, eps)
    if key in _cache: return _cache[key]
    if axis == "rz": subcircuit = Rz(angle, eps)
    elif axis == "ry": subcircuit = Ry(angle, eps)
    else: subcircuit = Rx(angle, eps)
    gate = subcircuit.to_gate()
    tc = count_t_gates_manual(dumps3(subcircuit))
    _cache[key] = (gate, tc)
    return gate, tc

def build_circuit(eps_list):
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

def total_t_count(eps_list):
    total = 0
    for j, idx in enumerate(rotation_indices):
        axis, angle = ops[idx][0], ops[idx][2]
        _, tc = _synthesize(axis, angle, eps_list[j])
        total += tc
    return total

def operator_distance(actual, reference):
    d = actual.shape[0]
    inner = np.trace(actual.conj().T @ reference)
    return float(np.sqrt(max(2 * d - 2 * np.abs(inner), 0.0)))

# --- SEQUENTIAL MULTI-TARGETING CONFIG ---
# More target distances for comprehensive exploration
TARGET_DISTANCES = [1e-1, 7e-2, 5e-2, 3e-2, 2e-2, 1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3]

# Ultra-granular epsilon sweep: 100 points from 1e-1 to 1e-9
EPS_COARSE_SWEEP = np.logspace(-1, -9, 100)

# More relaxation factors for fine-tuned optimization
RELAXATION_FACTORS = [100, 50, 30, 20, 15, 10, 7, 5, 3, 2, 1.5, 1.3, 1.2, 1.1, 1.05, 1.02]

if not os.path.exists("qasm"):
    os.makedirs("qasm")

print(f"Rotations to optimize: {n_rotations}")

# Main loop for multi-targeting
for TARGET_DIST in TARGET_DISTANCES:
    print(f"\n" + "="*60)
    print(f"RUNNING OPTIMIZATION FOR TARGET DISTANCE: {TARGET_DIST}")
    print("="*60)

    # Phase 1: Uniform Sweep
    phase1_hit = False
    best_uniform_eps = None
    for eps in EPS_COARSE_SWEEP:
        eps_list = [eps] * n_rotations
        qc = build_circuit(eps_list)
        dist = operator_distance(Operator(qc).data, target)
        if dist < TARGET_DIST:
            best_uniform_eps = eps
            phase1_hit = True
            break 

    if not phase1_hit:
        print(f"Warning: Target distance {TARGET_DIST} not reachable in sweep.")
        continue

    # Phase 2: Per-rotation relaxation
    current_eps = [best_uniform_eps] * n_rotations
    current_t = total_t_count(current_eps)
    current_dist = operator_distance(Operator(build_circuit(current_eps)).data, target)

    iteration = 0
    while True:
        iteration += 1
        any_improved = False
        for j in range(n_rotations):
            idx = rotation_indices[j]
            axis, angle, orig_eps = ops[idx][0], ops[idx][2], current_eps[j]
            _, orig_tc_j = _synthesize(axis, angle, orig_eps)

            for factor in RELAXATION_FACTORS:
                trial_eps = orig_eps * factor
                if trial_eps > 0.5: continue
                
                _, trial_tc_j = _synthesize(axis, angle, trial_eps)
                if trial_tc_j >= orig_tc_j: continue 

                trial_eps_list = current_eps.copy()
                trial_eps_list[j] = trial_eps
                trial_dist = operator_distance(Operator(build_circuit(trial_eps_list)).data, target)

                if trial_dist < TARGET_DIST:
                    current_t = current_t - orig_tc_j + trial_tc_j
                    current_eps[j] = trial_eps
                    current_dist = trial_dist
                    any_improved = True
                    break
        if not any_improved: break

    # Finalize result for this target
    final_qc = build_circuit(current_eps)
    final_t = total_t_count(current_eps)
    final_dist = operator_distance(Operator(final_qc).data, target)

    print(f"DONE -> Target: {TARGET_DIST} | Final T: {final_t} | Final Dist: {final_dist:.6e}")


print("\nAll target optimizations complete.")