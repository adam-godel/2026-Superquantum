from qiskit.qasm3 import dumps as dumps3
from utils import Rz
import math

qc = Rz(math.pi / 128, 1e-10)

qasm3_str = dumps3(qc)
with open("unitary2.qasm", 'w') as file:
    file.write(qasm3_str)

print("\nQASM3 file saved as unitary2.qasm")