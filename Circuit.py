import stim
import time
import numpy as np
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
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

def build_5qubit_code_circuit(p_error):
    circuit = stim.Circuit()
    
    # Initialize logical |0⟩
    circuit.append("H", [0])
    circuit.append("CNOT", [0, 1])
    circuit.append("CNOT", [0, 2])
    circuit.append("CNOT", [0, 3])
    circuit.append("CNOT", [0, 4])
    
    # Apply physical errors (more realistic than Pauli channel)
    for q in range(5):
        if random.random() < p_error:
            circuit.append("X", [q])
        if random.random() < p_error:
            circuit.append("Z", [q])
    
    # Measure stabilizers with explicit recording
    # XZZXI (ancilla 5 -> bit 0)
    circuit.append("H", [5])
    circuit.append("CNOT", [0, 5])  # X
    circuit.append("CZ", [1, 5])    # Z
    circuit.append("CZ", [2, 5])    # Z
    circuit.append("CNOT", [3, 5])  # X
    circuit.append("H", [5])
    circuit.append("M", [5])
    circuit.append("DETECTOR", [stim.target_rec(-1)])
    
    # IXZZX (ancilla 6 -> bit 1)
    circuit.append("H", [6])
    circuit.append("CNOT", [1, 6])  # X
    circuit.append("CZ", [2, 6])    # Z
    circuit.append("CZ", [3, 6])    # Z
    circuit.append("CNOT", [4, 6])  # X
    circuit.append("H", [6])
    circuit.append("M", [6])
    circuit.append("DETECTOR", [stim.target_rec(-1)])
    
    # XIXZZ (ancilla 7 -> bit 2)
    circuit.append("H", [7])
    circuit.append("CNOT", [0, 7])  # X
    circuit.append("CZ", [2, 7])    # Z
    circuit.append("CZ", [3, 7])    # Z
    circuit.append("CNOT", [4, 7])  # X
    circuit.append("H", [7])
    circuit.append("M", [7])
    circuit.append("DETECTOR", [stim.target_rec(-1)])
    
    # ZXIXZ (ancilla 8 -> bit 3)
    circuit.append("H", [8])
    circuit.append("CZ", [0, 8])    # Z
    circuit.append("CNOT", [1, 8])  # X
    circuit.append("CNOT", [3, 8])  # X
    circuit.append("CZ", [4, 8])    # Z
    circuit.append("H", [8])
    circuit.append("M", [8])
    circuit.append("DETECTOR", [stim.target_rec(-1)])
    
    return circuit

def simulate_5qubit_trial(p_error):
    circuit = build_5qubit_code_circuit(p_error)
    
    # Get syndrome in correct S1-S4 order
    measurements = circuit.compile_sampler().sample(1)[0]
    syndrome = [measurements[3], measurements[2], measurements[1], measurements[0]]
    
    
    # Apply corrections
    correction = stim.Circuit()
    qubit_corrections = decode_5qubit_syndrome(syndrome, p_error)
    
    for q, (x_flip, z_flip) in enumerate(qubit_corrections[:5]):
        if x_flip:
            correction.append("X", [q])
        if z_flip:
            correction.append("Z", [q])
    
    # Final measurement
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
        #if i % 1000 == 0:  # Print every 1000 trials
           # print(f"Completed {i}/{trials} trials (p={p_error})")
    return successes / trials

def run_5bit_experiment(p_error, trials=1000):
    start_time = time.time()
    successes = 0
    for i in range(trials):
        if simulate_5qubit_trial(p_error):
            successes += 1
        #if i % 100 == 0:  # Print progress every 100 trials
           # print(f"Completed {i}/{trials} trials (p={p_error})")
    runtime = time.time() - start_time
    return successes/trials, runtime

def build_5qubit_qiskit_circuit(p_error):
    """Matches Stim's circuit structure exactly"""
    qc = QuantumCircuit(9, 4)  # 5 data + 4 ancilla qubits
    
    # Initialize logical |0⟩ (same as Stim)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(0, 4)
    
    # Apply depolarizing noise (physical error model)
    for q in range(5):
        if random.random() < p_error:
            qc.x(q)
        if random.random() < p_error:
            qc.z(q)
    
    # Stabilizer measurements (same as Stim)
    # XZZXI (ancilla 5)
    qc.h(5)
    qc.cx(0, 5)
    qc.cz(1, 5)
    qc.cz(2, 5)
    qc.cx(3, 5)
    qc.h(5)
    qc.measure(5, 0)
    
    # IXZZX (ancilla 6)
    qc.h(6)
    qc.cx(1, 6)
    qc.cz(2, 6)
    qc.cz(3, 6)
    qc.cx(4, 6)
    qc.h(6)
    qc.measure(6, 1)
    
    # XIXZZ (ancilla 7)
    qc.h(7)
    qc.cx(0, 7)
    qc.cz(2, 7)
    qc.cz(3, 7)
    qc.cx(4, 7)
    qc.h(7)
    qc.measure(7, 2)
    
    # ZXIXZ (ancilla 8)
    qc.h(8)
    qc.cz(0, 8)
    qc.cx(1, 8)
    qc.cx(3, 8)
    qc.cz(4, 8)
    qc.h(8)
    qc.measure(8, 3)
    
    return qc

def decode_qiskit_syndrome(syndrome, p_error):
    """Wrapper for Stim decoder that returns separate X/Z corrections"""
    qubit_corrections = decode_5qubit_syndrome(syndrome, p_error)
    return (
        [x for x,z in qubit_corrections],
        [z for x,z in qubit_corrections]
    )

def run_stim_simulation(p_error, trials):
    start = time.time()
    successes = 0
    for _ in range(trials):
        circuit = build_5qubit_code_circuit(p_error)
    
        # Get syndrome in correct S1-S4 order
        measurements = circuit.compile_sampler().sample(1)[0]
        syndrome = [measurements[3], measurements[2], measurements[1], measurements[0]]
    
    
        # Apply corrections
        correction = stim.Circuit()
        qubit_corrections = decode_5qubit_syndrome(syndrome, p_error)
    
        for q, (x_flip, z_flip) in enumerate(qubit_corrections[:5]):
            if x_flip:
                correction.append("X", [q])
            if z_flip:
                correction.append("Z", [q])
    
        # Final measurement
        correction.append("H", [0])
        correction.append("CNOT", [0, 1])
        correction.append("CNOT", [0, 2])
        correction.append("CNOT", [0, 3])
        correction.append("CNOT", [0, 4])
        correction.append("M", [0])
    
        full_circuit = circuit + correction
        measurement = full_circuit.compile_sampler().sample(1)[0][-1]
        
        if not measurement:  # Logical |0⟩
            successes += 1

    runtime = time.time() - start
    return successes/trials, runtime

def run_qiskit_simulation(p_error, trials):
    simulator = AerSimulator()
    start = time.time()
    successes = 0
    
    for _ in range(trials):
        qc = build_5qubit_qiskit_circuit(p_error)
        
        # Get syndrome (same as Stim)
        result = simulator.run(qc, shots=1).result()
        syndrome = list(result.get_counts().keys())[0][::-1][:4]
        syndrome = [int(b) for b in syndrome]
        
        # Use Stim's MWPM decoder (same weights)
        x_corr, z_corr = decode_qiskit_syndrome(syndrome, p_error)
        
        # Apply corrections physically
        correction_qc = QuantumCircuit(9, 1)
        for q in range(5):
            if x_corr[q]:
                correction_qc.x(q)
            if z_corr[q]:
                correction_qc.z(q)
        
        # Measure logical qubit (same as Stim)
        correction_qc.h(0)
        correction_qc.cx(0, 1)
        correction_qc.cx(0, 2)
        correction_qc.cx(0, 3)
        correction_qc.cx(0, 4)
        correction_qc.measure(0, 0)
        
        # Run full circuit
        full_qc = qc.compose(correction_qc)
        result = simulator.run(full_qc, shots=1).result()
        measurement = int(list(result.get_counts().keys())[0][0])
        
        if not measurement:  # Logical |0⟩
            successes += 1
    
    runtime = time.time() - start
    return successes/trials, runtime



def benchmark():
    p_values = [0.01, 0.05, 0.1, 0.15, 0.2]
    trials = 100
    
    print(f"{'p_error':<10}{'Stim LR':<15}{'Stim Time':<15}{'Qiskit LR':<15}{'Qiskit Time':<15}")
    print("-"*60)
    
    stim_results = []
    qiskit_results = []
    
    for p in p_values:
        # Run Stim
        stim_lr, stim_time = run_5bit_experiment(p, trials)
        stim_results.append((stim_lr, stim_time))
        
        # Run Qiskit
        qiskit_lr, qiskit_time = run_qiskit_simulation(p, trials)
        qiskit_results.append((qiskit_lr, qiskit_time))
        
        print(f"{p:<10.3f}{stim_lr:<15.4f}{stim_time:<15.2f}{qiskit_lr:<15.4f}{qiskit_time:<15.2f}")

    # Plot results
    plt.figure(figsize=(12,5))
    
    # Error rates
    plt.subplot(121)
    plt.plot(p_values, [r[0] for r in stim_results], 'o-', label='Stim')
    plt.plot(p_values, [r[0] for r in qiskit_results], 's--', label='Qiskit')
    plt.xlabel("Physical Error Rate")
    plt.ylabel("Logical Success Rate")
    plt.legend()
    
    # Runtimes
    plt.subplot(122)
    plt.bar(['Stim (1000)', 'Qiskit (100)'], 
            [stim_results[0][1], qiskit_results[0][1]])
    plt.ylabel("Runtime (s)")
    plt.title("Performance Comparison")
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.show()
    
    


if __name__ == "__main__":
    #p_values = [0.01, 0.05, 0.1, 0.15, 0.2]
    #p_values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]  # More detail where it matters
    
    #results = {p: run_5bit_experiment(p, 10_000) for p in p_values}

    # Plotting
    #plt.figure()
    #plt.plot(list(results.keys()), list(results.values()), marker='o')
    #plt.xlabel("Physical Bit-Flip Probability (p)")
    #plt.ylabel("Logical Error Rate")
    #plt.title("5-Qubit Bit-Flip Code: Logical vs Physical Error Rate")
    #plt.grid(True)
    #plt.show()
    benchmark()
