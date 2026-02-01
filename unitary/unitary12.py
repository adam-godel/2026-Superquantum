import json
import numpy as np
from functools import reduce
from rmsynth.core import coeffs_to_vec, synthesize_from_coeffs
from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as dumps3

# ============================================================
# Read input
# ============================================================
with open('../challenge12.json') as f:
    data = json.load(f)

n = data['n']  # 9
terms = data['terms']
m = len(terms)  # 255
pauli_strings = [t['pauli'] for t in terms]
ks = [t['k'] for t in terms]

# ============================================================
# Pauli matrices
# ============================================================
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
_pauli = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}

# ============================================================
# Conjugation of Pauli strings by Clifford gates
# ============================================================
# CNOT(ctrl,tgt) conjugation table: (P_ctrl, P_tgt) -> (new_ctrl, new_tgt, sign)
# Derived from: CNOT(Pc⊗Pt)CNOT† = [CNOT(Pc⊗I)CNOT†][CNOT(I⊗Pt)CNOT†]
# with rules: X_c→X_c X_t, Z_c→Z_c, X_t→X_t, Z_t→Z_c Z_t
CNOT_TABLE = {
    ('I','I'):('I','I', 1), ('I','X'):('I','X', 1),
    ('I','Y'):('Z','Y', 1), ('I','Z'):('Z','Z', 1),
    ('X','I'):('X','X', 1), ('X','X'):('X','I', 1),
    ('X','Y'):('Y','Z', 1), ('X','Z'):('Y','Y',-1),
    ('Y','I'):('Y','X', 1), ('Y','X'):('Y','I', 1),
    ('Y','Y'):('X','Z',-1), ('Y','Z'):('X','Y', 1),
    ('Z','I'):('Z','I', 1), ('Z','X'):('Z','X', 1),
    ('Z','Y'):('I','Y', 1), ('Z','Z'):('I','Z', 1),
}

def apply_H(chars, signs, q):
    """H: X<->Z, Y->-Y"""
    for j in range(len(chars)):
        c = chars[j][q]
        if   c == 'X': chars[j][q] = 'Z'
        elif c == 'Z': chars[j][q] = 'X'
        elif c == 'Y': signs[j] *= -1

def apply_Sdg(chars, signs, q):
    """S†: X->-Y, Y->X, Z->Z, I->I"""
    for j in range(len(chars)):
        c = chars[j][q]
        if   c == 'X': chars[j][q] = 'Y'; signs[j] *= -1
        elif c == 'Y': chars[j][q] = 'X'

def apply_CX(chars, signs, ctrl, tgt):
    """CNOT(ctrl, tgt) conjugation."""
    for j in range(len(chars)):
        nc, nt, s = CNOT_TABLE[(chars[j][ctrl], chars[j][tgt])]
        chars[j][ctrl] = nc
        chars[j][tgt]  = nt
        signs[j] *= s

# ============================================================
# Gaussian elimination to find independent generators
# ============================================================
def find_basis(pauli_strings, n_qubits):
    """Row-reduce the symplectic matrix to find independent generators."""
    m = len(pauli_strings)
    M = np.zeros((m, 2*n_qubits), dtype=int)
    for j, p in enumerate(pauli_strings):
        for i, c in enumerate(p):
            if c in ('X','Y'): M[j, i]          = 1
            if c in ('Z','Y'): M[j, i+n_qubits] = 1
    indices = list(range(m))
    current = 0
    for col in range(2*n_qubits):
        pivot = -1
        for row in range(current, m):
            if M[row, col] == 1:
                pivot = row; break
        if pivot == -1:
            continue
        if pivot != current:
            M[[current, pivot]] = M[[pivot, current]]
            indices[current], indices[pivot] = indices[pivot], indices[current]
        for row in range(m):
            if row != current and M[row, col] == 1:
                M[row] = (M[row] + M[current]) % 2
        current += 1
    return indices[:current], current

# ============================================================
# Simultaneous diagonalization algorithm
# ============================================================
def find_diag_gates(pauli_strings, n_qubits):
    """
    Find Clifford gates that simultaneously diagonalize all commuting Paulis.
    Uses internal row operations to guide gate selection.
    Returns only the gate list (no row-op side effects leak out).
    """
    m = len(pauli_strings)
    chars = [list(p) for p in pauli_strings]
    signs = [1] * m

    basis_indices, rank = find_basis(pauli_strings, n_qubits)
    print(f"  Symplectic rank: {rank}")

    gates = []
    diag_info = []  # (target_qubit, row_index) for already-diagonalized generators

    for gen_idx in basis_indices:
        # --- Row operations: clear already-used qubits ---
        # By commutativity, current generator can only have I or Z at used qubits.
        for (used_q, diag_idx) in diag_info:
            ch = chars[gen_idx][used_q]
            assert ch in ('I', 'Z'), \
                f"Commutativity violated: gen {gen_idx} has {ch} at used qubit {used_q}"
            if ch == 'Z':
                # Multiply by diag generator (Z on used_q): Z*Z = I
                chars[gen_idx][used_q] = 'I'
                signs[gen_idx] *= signs[diag_idx]

        # Find remaining non-I qubits
        non_I = [i for i in range(n_qubits) if chars[gen_idx][i] != 'I']
        if not non_I:
            continue

        # --- Single-qubit gates: map X->Z and Y->Z ---
        for q in non_I:
            c = chars[gen_idx][q]
            if c == 'X':
                gates.append(('H', q))
                apply_H(chars, signs, q)
            elif c == 'Y':
                # S†: Y->X, then H: X->Z
                gates.append(('Sdg', q))
                apply_Sdg(chars, signs, q)
                gates.append(('H', q))
                apply_H(chars, signs, q)
            # Z: no gate needed

        # --- CNOT ladder: collect all Z onto one target qubit ---
        z_qubits = [i for i in range(n_qubits) if chars[gen_idx][i] == 'Z']
        target = z_qubits[0]
        for q in z_qubits[1:]:
            # CNOT(q, target) absorbs Z_q into Z_target
            gates.append(('CX', q, target))
            apply_CX(chars, signs, q, target)

        diag_info.append((target, gen_idx))

    return gates

def apply_clifford(pauli_strings, gates):
    """Apply Clifford gate list to Pauli strings. No row operations."""
    m = len(pauli_strings)
    chars = [list(p) for p in pauli_strings]
    signs = [1] * m
    for g in gates:
        if   g[0] == 'H':   apply_H(chars, signs, g[1])
        elif g[0] == 'Sdg': apply_Sdg(chars, signs, g[1])
        elif g[0] == 'CX':  apply_CX(chars, signs, g[1], g[2])
    return chars, signs

# ============================================================
# Main
# ============================================================
print("=== Challenge 12: Commuting Pauli Phase Program ===\n")

# Step 1: Find Clifford diagonalizing gates
print("Step 1: Finding simultaneous diagonalizing Clifford...")
gates = find_diag_gates(pauli_strings, n)
print(f"  Clifford circuit: {len(gates)} gates")

# Step 2: Apply Clifford to original Paulis to get true Z-strings and signs
print("Step 2: Computing Z-strings and signs...")
final_chars, final_signs = apply_clifford(pauli_strings, gates)

# Verify all strings are I/Z
for j in range(m):
    for i in range(n):
        assert final_chars[j][i] in ('I', 'Z'), \
            f"Term {j} ({pauli_strings[j]}) has {final_chars[j][i]} at qubit {i}: {''.join(final_chars[j])}"
print("  ✓ All 255 terms are I/Z after Clifford")

# Step 3: Compute phase polynomial coefficients
print("Step 3: Computing phase polynomial...")
phase_coeffs = {}
for j in range(m):
    support = sum(1 << i for i in range(n) if final_chars[j][i] == 'Z')
    assert support > 0, f"Term {j} mapped to identity"
    coeff = (ks[j] * final_signs[j]) % 8
    assert support not in phase_coeffs, f"Duplicate support {support:09b} from terms"
    phase_coeffs[support] = coeff

n1 = sum(1 for v in phase_coeffs.values() if v == 1)
n7 = sum(1 for v in phase_coeffs.values() if v == 7)
print(f"  {len(phase_coeffs)} terms: {n1} T-gates (coeff=1), {n7} T†-gates (coeff=7)")

# Step 4: Synthesize CNOT-optimised phase polynomial circuit via rmsynth
# The Optimizer changes coefficients (merges T→S) which is counterproductive
# for our {H,T,T†,CNOT}-only gate set.  Instead we synthesize directly from
# the original coefficients — all 1 or 7 (one T/T† each) — and let rmsynth
# choose an efficient CNOT layout via its scheduler.
print("Step 4: Synthesizing CNOT-optimised phase polynomial...")
vec = coeffs_to_vec(phase_coeffs, n)
synth_circ = synthesize_from_coeffs(vec, n, use_schedule=True)
print(f"  Phase polynomial T-count: {synth_circ.t_count()}")
print(f"  Phase polynomial CNOT count: {sum(1 for op in synth_circ.ops if op.kind=='cnot')}")

# Step 7: Build full Qiskit circuit: C → D → C†
print("Step 7: Building Qiskit circuit (C + phase_poly + C†)...")
qc = QuantumCircuit(n)

# --- Forward Clifford (C) ---
for g in gates:
    if   g[0] == 'H':   qc.h(g[1])
    elif g[0] == 'Sdg':
        qc.tdg(g[1]); qc.tdg(g[1])   # S† = T†²
    elif g[0] == 'CX':  qc.cx(g[1], g[2])

# --- Phase polynomial (D) directly from synthesised rmsynth circuit ---
# Each phase op has k ∈ {1,7}: emit 1 T or 1 T† respectively.
for op in synth_circ.ops:
    if op.kind == 'cnot':
        qc.cx(op.ctrl, op.tgt)
    elif op.kind == 'phase':
        k = op.k % 8
        if k == 0:
            continue
        # All coefficients are 1 or 7, so k is always 1 or 7 here.
        # Decompose into T/T† without S: min(k, 8-k) gates.
        if k <= 4:
            for _ in range(k):     qc.t(op.q)
        else:
            for _ in range(8 - k): qc.tdg(op.q)

# --- Inverse Clifford (C†): reverse order, invert each gate ---
for g in reversed(gates):
    if   g[0] == 'H':   qc.h(g[1])          # H† = H
    elif g[0] == 'Sdg':
        qc.t(g[1]); qc.t(g[1])              # (S†)† = S = T²
    elif g[0] == 'CX':  qc.cx(g[1], g[2])   # CX† = CX

# --- T-gate count summary ---
n_sdg = sum(1 for g in gates if g[0] == 'Sdg')
t_clifford = 4 * n_sdg   # 2 T† forward + 2 T inverse per Sdg
t_poly = synth_circ.t_count()  # all phase coeffs are 1 or 7 → 1 T each
print(f"  T from Clifford: {t_clifford} ({n_sdg} Sdg gates × 4)")
print(f"  T from phase poly: {t_poly}")
print(f"  Total T-gate count: {t_clifford + t_poly}")

# Step 8: Save QASM
print("Step 8: Saving QASM...")
qasm3_str = dumps3(qc)
with open('../qasm/unitary12.qasm', 'w') as f:
    f.write(qasm3_str)
print("  Saved qasm/unitary12.qasm")

# ============================================================
# Verification
# ============================================================
print("\n=== Verification ===")
from qiskit.quantum_info import Operator

U_circuit = Operator(qc).data

# Compute target unitary: U = ∏ exp(-i π k/8 P)
# Using exp(-iθP) = cos(θ)I - i·sin(θ)P  (since P²=I)
print("Computing target unitary...")
dim = 2**n
U_target = np.eye(dim, dtype=complex)
for term in terms:
    P = reduce(np.kron, [_pauli[ch] for ch in reversed(term['pauli'])])
    theta = np.pi * term['k'] / 8
    U_target = (np.cos(theta) * np.eye(dim) - 1j * np.sin(theta) * P) @ U_target

# Global phase alignment via trace inner product
tr = np.trace(U_circuit.conj().T @ U_target)
phase = tr / abs(tr) if abs(tr) > 1e-12 else 1.0
U_aligned = phase * U_circuit

err = np.linalg.norm(U_aligned - U_target) / np.sqrt(dim)
print(f"Normalized Frobenius distance: {err:.6e}")
if err < 1e-6:
    print("✓ Circuit matches target unitary!")
else:
    print("✗ Circuit does NOT match target unitary!")
    print(f"  Max element-wise error: {np.max(np.abs(U_aligned - U_target)):.6e}")
