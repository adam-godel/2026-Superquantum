from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as dumps3

qc = QuantumCircuit(2)
qc.t(1)
qc.t(1)
qc.cx(0, 1)
qc.tdg(1)
qc.tdg(1)

print(qc.draw())

qasm3_str = dumps3(qc)
with open("unitary2.qasm", 'w') as file:
    file.write(qasm3_str)