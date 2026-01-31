from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as dumps3

qc = QuantumCircuit(2)
qc.s(1)
qc.cx(0, 1)
qc.sdg(1)

qasm3_str = dumps3(qc)
with open("qasm/unitary1.qasm", 'w') as file:
    file.write(qasm3_str)