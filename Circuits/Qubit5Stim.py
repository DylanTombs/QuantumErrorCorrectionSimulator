import stim
import random

from Circuits.decoder import decode_5qubit_syndrome

def build_5qubit_stim_circuit(p_error):
    """Constructs the 5-qubit code circuit with noise and stabiliser measurements."""
    circuit = stim.Circuit()
    
    # Encode logical |0>
    circuit.append("H", [0])
    circuit.append("CX", [0, 1])
    circuit.append("CX", [0, 2])
    circuit.append("CX", [0, 3])
    circuit.append("CX", [0, 4])
    
    # Apply depolarizing noise (simplified as X and Z errors)
    for q in range(5):
        if random.random() < p_error:
            circuit.append("X", [q])
        if random.random() < p_error:
            circuit.append("Z", [q])
    
    # Measure stabilizers
    # XZZXI (ancilla 5 -> measurement record 0)
    circuit.append("H", [5])
    circuit.append("CX", [0, 5])
    circuit.append("CZ", [1, 5])
    circuit.append("CZ", [2, 5])
    circuit.append("CX", [3, 5])
    circuit.append("H", [5])
    circuit.append("M", [5])
    
    # IXZZX (ancilla 6 -> measurement record 1)
    circuit.append("H", [6])
    circuit.append("CX", [1, 6])
    circuit.append("CZ", [2, 6])
    circuit.append("CZ", [3, 6])
    circuit.append("CX", [4, 6])
    circuit.append("H", [6])
    circuit.append("M", [6])
    
    # XIXZZ (ancilla 7 -> measurement record 2)
    circuit.append("H", [7])
    circuit.append("CX", [0, 7])
    circuit.append("CZ", [2, 7])
    circuit.append("CZ", [3, 7])
    circuit.append("CX", [4, 7])
    circuit.append("H", [7])
    circuit.append("M", [7])
    
    # ZXIXZ (ancilla 8 -> measurement record 3)
    circuit.append("H", [8])
    circuit.append("CZ", [0, 8])
    circuit.append("CX", [1, 8])
    circuit.append("CX", [3, 8])
    circuit.append("CZ", [4, 8])
    circuit.append("H", [8])
    circuit.append("M", [8])
    
    return circuit

def decode_5qubit_syndrome_stim(syndrome, p_error):
    """Returns separate X/Z correction lists from MWPM decoding."""
    qubit_corrections = decode_5qubit_syndrome(syndrome, p_error)
    x_corr = [x for x, _ in qubit_corrections]
    z_corr = [z for _, z in qubit_corrections]
    return x_corr, z_corr