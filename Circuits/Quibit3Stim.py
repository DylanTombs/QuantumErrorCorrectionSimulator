# fast_stim_qec.py
import stim
import random

def build_3qubit_stim_circuit(p_error):
    """Constructs the 3-qubit repetition code circuit with noise and stabiliser measurements."""
    circuit = stim.Circuit()
    
    # Encode logical |0> using repetition code
    circuit.append("CX", [0, 1])  # Copy qubit 0 to qubit 1
    circuit.append("CX", [0, 2])  # Copy qubit 0 to qubit 2
    
    # Apply bit-flip noise to each data qubit
    for q in range(3):
        if random.random() < p_error:
            circuit.append("X", [q])
    
    # Measure stabilizers using ancilla qubits
    # Stabilizer 1: Z0*Z1 (parity of qubits 0 and 1)
    circuit.append("CX", [0, 3])  # Copy qubit 0 to ancilla 3
    circuit.append("CX", [1, 3])  # XOR with qubit 1
    circuit.append("M", [3])      # Measure ancilla 3 -> measurement record 0
    
    # Stabilizer 2: Z1*Z2 (parity of qubits 1 and 2)
    circuit.append("CX", [1, 4])  # Copy qubit 1 to ancilla 4
    circuit.append("CX", [2, 4])  # XOR with qubit 2
    circuit.append("M", [4])      # Measure ancilla 4 -> measurement record 1
    
    return circuit

def decode_3qubit_syndrome_stim(syndrome):
    """
    Decode syndrome for 3-qubit repetition code.
    syndrome[0] = Z0*Z1 measurement
    syndrome[1] = Z1*Z2 measurement
    Returns list of X corrections for each qubit.
    """
    corrections = [0, 0, 0]
    
    if syndrome == [1, 0]:
        corrections[0] = 1  # Correct qubit 0
    elif syndrome == [1, 1]:
        corrections[1] = 1  # Correct qubit 1
    elif syndrome == [0, 1]:
        corrections[2] = 1  # Correct qubit 2
    
    return corrections

