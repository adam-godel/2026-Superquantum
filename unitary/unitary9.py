from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as dumps3
import math

qc = QuantumCircuit(2)



qasm3_str = dumps3(qc)
with open("qasm/unitary9.qasm", 'w') as file:
    file.write(qasm3_str)