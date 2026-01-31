from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as dumps3
from utils import Ry
import math

theta = math.pi / 7
epsilon = 1.0e-01

qc = QuantumCircuit(2)
qc.append(Ry(theta/2, epsilon).to_gate(), [1])
qc.cx(0, 1)
qc.append(Ry(-theta/2, epsilon).to_gate(), [1])
qc.cx(0, 1)

qasm3_str = dumps3(qc)
with open("qasm/unitary2.qasm", 'w') as file:
    file.write(qasm3_str)