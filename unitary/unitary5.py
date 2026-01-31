from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as dumps3

qc = QuantumCircuit(2)
qc.cx(0, 1)
qc.cx(1, 0)
qc.cx(0, 1)

qasm3_str = dumps3(qc)
with open("qasm/unitary5.qasm", 'w') as file:
    file.write(qasm3_str)