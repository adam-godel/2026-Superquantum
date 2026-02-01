from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as dumps3
from utils import Rz
import math

theta = math.pi / 7
epsilon = 3.16e-10

qc = QuantumCircuit(2)

qc.h(0); qc.h(1)
qc.s(0); qc.s(1)
qc.h(0); qc.h(1)
qc.cx(0, 1)
qc.append(Rz(-2*theta, epsilon).to_gate(), [1])
qc.cx(0, 1)
qc.h(0); qc.h(1)
qc.sdg(0); qc.sdg(1)
qc.h(0); qc.h(1)

qc.h(0); qc.h(1)
qc.cx(0, 1)
qc.append(Rz(-2*theta, epsilon).to_gate(), [1])
qc.cx(0, 1)
qc.h(0); qc.h(1)

qasm3_str = dumps3(qc)
with open("qasm/unitary4.qasm", 'w') as file:
    file.write(qasm3_str)