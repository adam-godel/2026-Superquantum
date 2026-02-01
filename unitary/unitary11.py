from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as dumps3

from rmsynth import Circuit, Optimizer
from rmsynth.core import extract_phase_coeffs

circ = Circuit(4)
circ.add_phase(0, 1); circ.add_phase(1, 1); circ.add_phase(2, 1); circ.add_phase(3, 1)
circ.add_cnot(0, 1); circ.add_phase(1, 1); circ.add_cnot(0, 1)
circ.add_cnot(0, 2); circ.add_phase(2, 1); circ.add_cnot(0, 2)
circ.add_cnot(0, 3); circ.add_phase(3, 1); circ.add_cnot(0, 3)

circ.add_cnot(1, 2); circ.add_phase(2, 1); circ.add_cnot(1, 2)
circ.add_cnot(1, 3); circ.add_phase(3, 1); circ.add_cnot(1, 3)

circ.add_cnot(2, 3); circ.add_phase(3, 1); circ.add_cnot(2, 3)

circ.add_cnot(1, 3)
circ.add_cnot(2, 3)
circ.add_phase(3, 1)
circ.add_cnot(2, 3)
circ.add_cnot(1, 3)

opt = Optimizer(decoder="rpa", effort=3, policy="distance+depth", policy_lambda=5)
new_circ, rep = opt.optimize(circ)

n = new_circ.n
a = extract_phase_coeffs(new_circ)

qc = QuantumCircuit(4)

for key in range(1, 2**n):
    val = a.get(key, 0)
    if val == 0:
        continue
    qubits = [i for i in range(n) if key & (1 << i)]
    target = qubits[-1]
    controls = qubits[:-1]
    for c in controls:
        qc.cx(c, target)
    for _ in range(val // 2):
        qc.s(target)
    if val % 2:
        qc.t(target)
    for c in reversed(controls):
        qc.cx(c, target)

qasm3_str = dumps3(qc)
with open("qasm/unitary11.qasm", 'w') as file:
    file.write(qasm3_str)