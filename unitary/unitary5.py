from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as dumps3
from utils import Rz, Ry
import math

theta = math.pi / 7
epsilon = 1e-10

qc = QuantumCircuit(2)
qc.h(0); qc.h(1)
qc.h(1); qc.cx(0, 1); qc.h(1) # cz gate
qc.cx(0, 1); qc.cx(1, 0); qc.cx(0, 1) # swap gate
qc.append(Rz(-4*theta, epsilon).to_gate(), [0])
qc.cx(0, 1); qc.cx(1, 0); qc.cx(0, 1) # swap gate
qc.h(1); qc.cx(0, 1); qc.h(1) # cz gate
qc.h(0); qc.h(1)

qasm3_str = dumps3(qc)
with open("qasm/unitary5.qasm", 'w') as file:
    file.write(qasm3_str)