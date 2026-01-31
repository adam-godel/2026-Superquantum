from qiskit.qasm3 import dumps as dumps3
from utils import gates_to_qiskit_circuit, Rz
import math

Rz(math.pi / 128, 1e-10)

input_seq = "HTSHTSHTHTSHTSHTHTHTSHTHTSHTSHTHTSHTHTHTSHTHTHTHTSHTSHTSHTHTSHTSHTHTHTHTSHTSHTSHTSHTSHTHTSHTHTHTSHTSHTSHTHTSHTHTHTSHTSHTHTHTSHTSHTSHTSHTHTHTHTHTSHTSHTHTHTSHTHTSHTSHTSHTSHTSHTHTHTHTHTSHTHTHTSHTHTHTHTSHTSHTSHTHTSHTSHTSHTSHTSHTSHTHTSHTSHTSHTHTHTSHTSHTSHTSHTSHTSHXSS"

qc = gates_to_qiskit_circuit(input_seq, i=1, reverse=False)

# Export to QASM3
qasm3_str = dumps3(qc)
with open("unitary2.qasm", 'w') as file:
    file.write(qasm3_str)

print("\nQASM3 file saved as unitary2.qasm")