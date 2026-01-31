from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as dumps3
from utils import Rz, Ry
import math

theta = math.pi / 7
epsilon = 1e-10

qc = QuantumCircuit(2)
qc.append(Ry(theta, epsilon).to_gate(), [1])
qc.cx(0, 1)
qc.append(Rz(-theta, epsilon).to_gate(), [0])
qc.append(Ry(-theta, epsilon).to_gate(), [1])
qc.cx(0, 1)
qc.append(Ry(theta, epsilon).to_gate(), [1])

qasm3_str = dumps3(qc)
with open("qasm/unitary4.qasm", 'w') as file:
    file.write(qasm3_str)