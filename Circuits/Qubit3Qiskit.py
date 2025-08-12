import random
from qiskit import QuantumCircuit
from  qiskit_aer.noise import NoiseModel, depolarizing_error

def build_3qubit_qiskit_circuit(p_error):
    """Constructs the 3-qubit repetition code circuit with noise and stabilizer measurements."""
    qc = QuantumCircuit(5, 2)  # 3 data + 2 ancilla qubits, 2 classical bits for syndrome
    
    # Encode logical |0> using repetition code
    qc.cx(0, 1)  # Copy qubit 0 to qubit 1
    qc.cx(0, 2)  # Copy qubit 0 to qubit 2
    
    # Apply depolarizing noise (simplified - just bit flip errors for repetition code)
    for q in range(3):
        if random.random() < p_error:
            qc.x(q)  # Bit flip error
    
    # Measure stabilizers for 3-qubit repetition code
    # Stabilizer 1: Z0*Z1 (parity of qubits 0 and 1)
    qc.cx(0, 3)  # Copy qubit 0 to ancilla 3
    qc.cx(1, 3)  # XOR with qubit 1
    qc.measure(3, 0)  # Measure ancilla 3 to classical bit 0
    
    # Stabilizer 2: Z1*Z2 (parity of qubits 1 and 2)
    qc.cx(1, 4)  # Copy qubit 1 to ancilla 4
    qc.cx(2, 4)  # XOR with qubit 2
    qc.measure(4, 1)  # Measure ancilla 4 to classical bit 1
    
    return qc

def decode_3qubit_syndrome(syndrome):
    """
    Decode syndrome for 3-qubit repetition code.
    syndrome[0] = Z0*Z1 measurement
    syndrome[1] = Z1*Z2 measurement
    Returns list of X corrections for each qubit.
    """
    corrections = [0, 0, 0]  # No corrections initially
    
    # Syndrome interpretation for 3-qubit repetition code:
    # [0, 0] -> No error
    # [1, 0] -> Error on qubit 0
    # [1, 1] -> Error on qubit 1
    # [0, 1] -> Error on qubit 2
    
    if syndrome == [1, 0]:
        corrections[0] = 1  # Correct qubit 0
    elif syndrome == [1, 1]:
        corrections[1] = 1  # Correct qubit 1
    elif syndrome == [0, 1]:
        corrections[2] = 1  # Correct qubit 2
    
    return corrections

def build_noise_model(p_error):
    noise = NoiseModel()
    err = depolarizing_error(p_error, 1)
    for q in range(3):
        noise.add_all_qubit_quantum_error(err, ['id', 'x', 'y', 'z', 'cx', 'cz', 'h'])
    return noise

# Alternative implementation using majority vote for logical readout
def run_qiskit_3qubit_majority(p_error: float, trials: int, seed: int):
    """3-qubit code with majority vote readout (no syndrome correction)."""
    from qiskit_aer import AerSimulator
    import numpy as np
    import time
    
    np.random.seed(seed)
    random.seed(seed)
    simulator = AerSimulator()
    successes = 0
    
    start = time.perf_counter()
    for _ in range(trials):
        qc = QuantumCircuit(3, 3)
        
        # Encode logical |0>
        qc.cx(0, 1)
        qc.cx(0, 2)
        
        # Apply noise
        for q in range(3):
            if random.random() < p_error:
                qc.x(q)
        
        # Measure all qubits
        qc.measure_all()
        
        # Get results and apply majority vote
        result = simulator.run(qc, shots=1).result()
        measurements = [int(b) for b in list(result.get_counts().keys())[0][::-1]]
        
        # Majority vote
        logical_result = 1 if sum(measurements) > 1.5 else 0
        successes += (logical_result == 0)
    
    runtime = time.perf_counter() - start
    return {"success_rate": successes / trials, "runtime_s": runtime}
