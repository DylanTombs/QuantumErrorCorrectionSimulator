import stim
import time
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import Statevector

def build3QubitCodeCircuit(p_error):
    circuit = stim.Circuit()
    
    # Initialize logical |0⟩ state (3 physical qubits)
    circuit.append("H", [0])
    circuit.append("CNOT", [0, 1])
    circuit.append("CNOT", [0, 2])
    
    # Add possible errors
    circuit.append("DEPOLARIZE1", [0, 1, 2], p_error)
    
    # Error correction (measure stabilizers)
    circuit.append("CNOT", [0, 3])  # Ancilla qubit
    circuit.append("CNOT", [1, 3])
    circuit.append("M", [3])  # Measure Z1Z2
    
    circuit.append("CNOT", [0, 4])  # Another ancilla
    circuit.append("CNOT", [2, 4])
    circuit.append("M", [4])  # Measure Z1Z3

    return circuit

def build_5qubit_code_circuit(p_error):
    """Builds a 5-qubit code circuit with depolarizing noise."""
    circuit = stim.Circuit()
    
    # Data qubits (0-4), ancillas (5-8)
    
    # Apply depolarizing noise to data qubits
    for q in range(5):
        circuit.append("DEPOLARIZE1", [q], p_error)
    
    # Stabilizer measurements
    # S1 = X Z Z X I (ancilla 5)
    circuit.append("H", [5])
    circuit.append("CNOT", [0, 5])
    circuit.append("CZ", [1, 5])
    circuit.append("CZ", [2, 5])
    circuit.append("CNOT", [3, 5])
    circuit.append("H", [5])
    circuit.append("M", [5])
    
    # S2 = I X Z Z X (ancilla 6)
    circuit.append("H", [6])
    circuit.append("CNOT", [1, 6])
    circuit.append("CZ", [2, 6])
    circuit.append("CZ", [3, 6])
    circuit.append("CNOT", [4, 6])
    circuit.append("H", [6])
    circuit.append("M", [6])
    
    # S3 = X I X Z Z (ancilla 7)
    circuit.append("H", [7])
    circuit.append("CNOT", [0, 7])
    circuit.append("CZ", [2, 7])
    circuit.append("CZ", [3, 7])
    circuit.append("CNOT", [2, 7])  # Note: This should likely be [4,7] for Z Z?
    circuit.append("H", [7])
    circuit.append("M", [7])
    
    # S4 = Z X I X Z (ancilla 8)
    circuit.append("H", [8])
    circuit.append("CZ", [0, 8])
    circuit.append("CNOT", [1, 8])
    circuit.append("CNOT", [3, 8])
    circuit.append("CZ", [4, 8])
    circuit.append("H", [8])
    circuit.append("M", [8])
    
    return circuit

def simulate_3qubit_trial(p_error: float):
    circuit = build3QubitCodeCircuit(p_error)

   
    sampler = circuit.compile_sampler()
    syndrome = sampler.sample(1)[0]  # Get measurements [Z1Z2, Z1Z3]
    
    correction_qubits = decode_syndrome_with_mwpm(tuple(syndrome))
    correction = stim.Circuit()
    for qubit, should_flip in enumerate(correction_qubits):
        if should_flip:
            correction.append("X", [qubit])
    # Combine circuits

    full_circuit = circuit + correction
    measurement = full_circuit.compile_sampler().sample(1)[0][-1]
    return not measurement

def decode_syndrome_with_mwpm(syndrome: tuple):
    s01, s02 = syndrome
    if s01 and not s02:
        return [0, 1, 0]
    elif not s01 and s02:
        return [0, 0, 1]
    elif s01 and s02:
        return [1, 0, 0]
    else:
        return [0, 0, 0]

def decode_5qubit_syndrome(syndrome, p_error):
    """MWPM decoder for 5-qubit code handling both X and Z errors."""
    # Split syndrome into X and Z parts (alternating bits)
    x_syndrome = syndrome[::2]  # Odd-indexed stabilizers (S1, S3)
    z_syndrome = syndrome[1::2] # Even-indexed stabilizers (S2, S4)
    
    x_correction = decode_x_syndrome(x_syndrome, p_error)
    z_correction = decode_z_syndrome(z_syndrome, p_error)
    
    # Combine corrections
    return [ (x,z) for x,z in zip(x_correction, z_correction) ]

def decode_x_syndrome(syndrome, p_error):
    """Decoder for X-type errors (using S1 and S3)."""
    G = nx.Graph()
    boundary = "BX"
    G.add_node(boundary)
    
    # Add detection events
    for i, s in enumerate(syndrome):
        if s:
            node = f"X{i+1}"  # X1, X3
            G.add_node(node)
            G.add_edge(node, boundary, weight=-math.log(p_error))
    
    # Add edges between X stabilizers
    stabilizer_edges = [
        ("X1", "X3", -math.log(p_error**2))  # Weight for two X errors
    ]
    
    for u, v, weight in stabilizer_edges:
        if u in G and v in G:
            G.add_edge(u, v, weight=weight)
    
    matching = nx.min_weight_matching(G)
    
    correction = [0]*5
    for pair in matching:
        if boundary in pair:
            continue
        
        s1, s2 = sorted(pair)
        if s1 == "X1" and s2 == "X3":
            # Most likely single X error on qubit 0
            correction[0] ^= 1
    
    return correction

def decode_z_syndrome(syndrome, p_error):
    """Decoder for Z-type errors (using S2 and S4)."""
    G = nx.Graph()
    boundary = "BZ"
    G.add_node(boundary)
    
    # Add detection events
    for i, s in enumerate(syndrome):
        if s:
            node = f"Z{i+2}"  # Z2, Z4
            G.add_node(node)
            G.add_edge(node, boundary, weight=-math.log(p_error))
    
    # Add edges between Z stabilizers
    stabilizer_edges = [
        ("Z2", "Z4", -math.log(p_error**2))  # Weight for two Z errors
    ]
    
    for u, v, weight in stabilizer_edges:
        if u in G and v in G:
            G.add_edge(u, v, weight=weight)
    
    matching = nx.min_weight_matching(G)
    
    correction = [0]*5
    for pair in matching:
        if boundary in pair:
            continue
        
        s1, s2 = sorted(pair)
        if s1 == "Z2" and s2 == "Z4":
            # Most likely single Z error on qubit 1
            correction[1] ^= 1
    
    return correction

def simulate_5qubit_trial(p_error):
    circuit = build_5qubit_code_circuit(p_error)
    sampler = circuit.compile_sampler()
    syndrome = sampler.sample(1)[0]  # Get all stabilizer measurements
    
    correction = stim.Circuit()
    qubit_corrections = decode_5qubit_syndrome(syndrome, p_error)
    
    # Apply both X and Z corrections
    for q, (x_flip, z_flip) in enumerate(qubit_corrections):
        if x_flip:
            correction.append("X", [q])
        if z_flip:
            correction.append("Z", [q])
    
    # Proper logical measurement
    correction.append("H", [0])
    correction.append("CNOT", [0, 1])
    correction.append("CNOT", [0, 2])
    correction.append("CNOT", [0, 3])
    correction.append("CNOT", [0, 4])
    correction.append("M", [0])
    
    full_circuit = circuit + correction
    measurement = full_circuit.compile_sampler().sample(1)[0][-1]
    return not measurement

def run_3bit_experiment(p_error: float, trials: int = 100_000):
    successes = 0
    for i in range(trials):
        if simulate_3qubit_trial(p_error):
            successes += 1
        if i % 1000 == 0:  # Print every 1000 trials
            print(f"Completed {i}/{trials} trials (p={p_error})")
    return successes / trials

def run_5bit_experiment(p_error, trials=1000):
    successes = 0
    for i in range(trials):
        if simulate_5qubit_trial(p_error):
            successes += 1
        if i % 1000 == 0:
            print(f"Completed {i}/{trials} trials (p={p_error})")
    return successes / trials


def build_5qubit_qiskit_circuit(p_error):
    qc = QuantumCircuit(9, 5)  # 5 data + 4 ancilla qubits
    
    # Initialize logical |0⟩
    qc.h(0)
    for i in range(1,5):
        qc.cx(0, i)
    
    # Apply depolarizing noise (simplified)
    for q in range(5):
        if np.random.random() < p_error:
            qc.x(q)
        if np.random.random() < p_error:
            qc.z(q)
    
    # Stabilizer measurements
    # ... (implement stabilizers using Qiskit gates) ...
    
    return qc

def run_qiskit_simulation(p_error, trials):
    backend = Aer.get_backend('statevector_simulator')
    start = time.time()
    successes = 0
    
    for _ in range(trials):
        qc = build_5qubit_qiskit_circuit(p_error)
        result = execute(qc, backend).result()
        state = Statevector(result.get_statevector())
        
        # Measure logical qubit
        if state.probabilities([0])[0] > 0.99:  # |0⟩ state
            successes += 1
    
    runtime = time.time() - start
    return successes/trials, runtime

def benchmark():
    p_values = np.linspace(0.01, 0.2, 5)
    trials = 1000
    
    print(f"{'p_error':<10}{'Stim LR':<15}{'Stim Time':<15}{'Qiskit LR':<15}{'Qiskit Time':<15}")
    print("-"*60)
    
    for p in p_values:
        # Run Stim
        stim_lr, stim_time = run_5bit_experiment(p, trials)
        
        # Run Qiskit (fewer trials due to slower speed)
        qiskit_lr, qiskit_time = run_qiskit_simulation(p, min(trials, 100))
        
        print(f"{p:<10.3f}{stim_lr:<15.4f}{stim_time:<15.2f}{qiskit_lr:<15.4f}{qiskit_time:<15.2f}")

    # Plot results
    plt.figure(figsize=(12,5))
    
    # Plot error rates
    plt.subplot(121)
    plt.plot(p_values, [run_5bit_experiment(p, 1000)[0] for p in p_values], 'o-', label='Stim')
    plt.plot(p_values, [run_qiskit_simulation(p, 100)[0] for p in p_values], 's--', label='Qiskit')
    plt.xlabel("Physical Error Rate")
    plt.ylabel("Logical Success Rate")
    plt.legend()
    
    # Plot runtimes
    plt.subplot(122)
    plt.bar(['Stim', 'Qiskit'], 
            [run_5bit_experiment(0.1, 1000)[1],
            [run_qiskit_simulation(0.1, 100)[1]]])
    plt.ylabel("Runtime (s)")
    plt.title("Performance Comparison")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #p_values = [0.01, 0.05, 0.1, 0.15, 0.2]
    p_values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]  # More detail where it matters
    
    results = {p: run_5bit_experiment(p, 10_000) for p in p_values}

    # Plotting
    plt.figure()
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel("Physical Bit-Flip Probability (p)")
    plt.ylabel("Logical Error Rate")
    plt.title("5-Qubit Bit-Flip Code: Logical vs Physical Error Rate")
    plt.grid(True)
    plt.show()
