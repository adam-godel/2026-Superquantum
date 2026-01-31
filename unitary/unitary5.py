from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as dumps3
from utils import Rz, Ry
import math

theta = math.pi / 7
epsilon = 1e-10

qc_zz = QuantumCircuit(2)
qc_zz.cx(0, 1)
qc_zz.append(Rz(2*theta, epsilon).to_gate(), [1])
qc_zz.cx(0, 1)

qc_xxyy = QuantumCircuit(2)
qc_xxyy.append(Ry(theta, epsilon).to_gate(), [1])
qc_xxyy.cx(0, 1)
qc_xxyy.append(Rz(-theta, epsilon).to_gate(), [0])
qc_xxyy.append(Ry(-theta, epsilon).to_gate(), [1])
qc_xxyy.cx(0, 1)
qc_xxyy.append(Ry(theta, epsilon).to_gate(), [1])

qasm3_str = dumps3(qc)
with open("qasm/unitary4.qasm", 'w') as file:
    file.write(qasm3_str)