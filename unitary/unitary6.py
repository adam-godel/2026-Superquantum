from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as dumps3
from utils import Rz
import math

theta = math.pi / 7
epsilon = 1e-8

qc = QuantumCircuit(2)

qc.h(0); qc.h(1)
qc.cx(0, 1)
qc.append(Rz(-2*theta, epsilon).to_gate(), [1])
qc.cx(0, 1)
qc.h(0); qc.h(1)

qc.append(Rz(-theta, epsilon).to_gate(), [0])
qc.append(Rz(-theta, epsilon).to_gate(), [1])

qasm3_str = dumps3(qc)
with open("qasm/unitary6.qasm", 'w') as file:
    file.write(qasm3_str)