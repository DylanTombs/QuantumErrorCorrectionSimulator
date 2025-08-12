import random
from qiskit import QuantumCircuit

from Circuits.decoder import decode_5qubit_syndrome


def build_5qubit_qiskit_circuit(p_error):
    """Constructs the 5-qubit code circuit with noise and stabiliser measurements."""
    qc = QuantumCircuit(9, 4)  # 5 data + 4 ancilla
    
    # Encode logical |0>
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)

    qc.cx(0, 4)
    
    # Apply depolarizing noise
    for q in range(5):
        if random.random() < p_error:
            qc.x(q)
        if random.random() < p_error:
            qc.z(q)
    
    # Measure stabilisers
    # XZZXI  (ancilla 5 -> bit 0)
    qc.h(5)
    qc.cx(0, 5)
    qc.cz(1, 5)
    qc.cz(2, 5)
    qc.cx(3, 5)
    qc.h(5)
    qc.measure(5, 0)
    
    # IXZZX  (ancilla 6 -> bit 1)
    qc.h(6)
    qc.cx(1, 6)
    qc.cz(2, 6)
    qc.cz(3, 6)
    qc.cx(4, 6)
    qc.h(6)
    qc.measure(6, 1)
    
    # XIXZZ  (ancilla 7 -> bit 2)
    qc.h(7)
    qc.cx(0, 7)
    qc.cz(2, 7)
    qc.cz(3, 7)
    qc.cx(4, 7)
    qc.h(7)
    qc.measure(7, 2)
    
    # ZXIXZ  (ancilla 8 -> bit 3)
    qc.h(8)
    qc.cz(0, 8)
    qc.cx(1, 8)
    qc.cx(3, 8)
    qc.cz(4, 8)
    qc.h(8)
    qc.measure(8, 3)
    
    return qc

def decode_qiskit_syndrome(syndrome, p_error):
    """Returns separate X/Z correction lists from MWPM decoding."""
    qubit_corrections = decode_5qubit_syndrome(syndrome, p_error)
    x_corr = [x for x, _ in qubit_corrections]
    z_corr = [z for _, z in qubit_corrections]
    return x_corr, z_corr
