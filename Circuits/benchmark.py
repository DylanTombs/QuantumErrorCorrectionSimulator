import math
import random
import time
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# ---------------------------
# MWPM DECODER IMPLEMENTATION
# ---------------------------

def decode_5qubit_syndrome(syndrome, p_error):
    """MWPM decoder for 5-qubit code handling both X and Z errors."""
    # Split syndrome into X and Z parts (alternating bits)
    x_syndrome = syndrome[::2]  # S1, S3
    z_syndrome = syndrome[1::2] # S2, S4
    
    x_correction = decode_x_syndrome(x_syndrome, p_error)
    z_correction = decode_z_syndrome(z_syndrome, p_error)
    
    # Combine into [(x0,z0), (x1,z1), ...]
    return [(x, z) for x, z in zip(x_correction, z_correction)]

def decode_x_syndrome(syndrome, p_error):
    """Decoder for X-type errors using stabilizers S1 and S3."""
    G = nx.Graph()
    boundary = "BX"
    G.add_node(boundary)
    
    # Detection events
    for i, s in enumerate(syndrome):
        if s:
            node = f"X{i+1}"
            G.add_node(node)
            G.add_edge(node, boundary, weight=-math.log(p_error))
    
    # Possible error links
    stabilizer_edges = [("X1", "X3", -math.log(p_error**2))]
    for u, v, w in stabilizer_edges:
        if u in G and v in G:
            G.add_edge(u, v, weight=w)
    
    matching = nx.min_weight_matching(G)
    correction = [0] * 5
    
    for pair in matching:
        if boundary in pair:
            continue
        s1, s2 = sorted(pair)
        if s1 == "X1" and s2 == "X3":
            correction[0] ^= 1  # flip qubit 0
    return correction

def decode_z_syndrome(syndrome, p_error):
    """Decoder for Z-type errors using stabilizers S2 and S4."""
    G = nx.Graph()
    boundary = "BZ"
    G.add_node(boundary)
    
    # Detection events
    for i, s in enumerate(syndrome):
        if s:
            node = f"Z{i+2}"
            G.add_node(node)
            G.add_edge(node, boundary, weight=-math.log(p_error))
    
    # Possible error links
    stabilizer_edges = [("Z2", "Z4", -math.log(p_error**2))]
    for u, v, w in stabilizer_edges:
        if u in G and v in G:
            G.add_edge(u, v, weight=w)
    
    matching = nx.min_weight_matching(G)
    correction = [0] * 5
    
    for pair in matching:
        if boundary in pair:
            continue
        s1, s2 = sorted(pair)
        if s1 == "Z2" and s2 == "Z4":
            correction[1] ^= 1  # flip qubit 1
    return correction

# ---------------------------
# CIRCUIT CONSTRUCTION
# ---------------------------

def build_5qubit_qiskit_circuit(p_error):
    """Constructs the 5-qubit code circuit with noise and stabilizer measurements."""
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
    
    # Measure stabilizers
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

# ---------------------------
# DECODER WRAPPER
# ---------------------------

def decode_qiskit_syndrome(syndrome, p_error):
    """Returns separate X/Z correction lists from MWPM decoding."""
    qubit_corrections = decode_5qubit_syndrome(syndrome, p_error)
    x_corr = [x for x, _ in qubit_corrections]
    z_corr = [z for _, z in qubit_corrections]
    return x_corr, z_corr

# ---------------------------
# SIMULATION
# ---------------------------

def run_qiskit_simulation(p_error, trials):
    simulator = AerSimulator()
    start = time.time()
    successes = 0
    
    for _ in range(trials):
        # Build noisy encoded circuit
        qc = build_5qubit_qiskit_circuit(p_error)
        
        # Measure stabilizers to get syndrome
        result = simulator.run(qc, shots=1).result()
        syndrome_bits = list(result.get_counts().keys())[0][::-1][:4]
        syndrome = [int(b) for b in syndrome_bits]
        
        # Decode and get corrections
        x_corr, z_corr = decode_qiskit_syndrome(syndrome, p_error)
        
        # Apply corrections + logical measurement
        correction_qc = QuantumCircuit(9, 1)
        for q in range(5):
            if x_corr[q]:
                correction_qc.x(q)
            if z_corr[q]:
                correction_qc.z(q)
        
        # Decode logical qubit back to |0>
        correction_qc.h(0)
        correction_qc.cx(0, 1)
        correction_qc.cx(0, 2)
        correction_qc.cx(0, 3)
        correction_qc.cx(0, 4)
        correction_qc.measure(0, 0)
        
        # Run full circuit (encode + noise + syndrome + corrections + decode)
        full_qc = qc.compose(correction_qc)
        result = simulator.run(full_qc, shots=1).result()
        meas = int(list(result.get_counts().keys())[0][0])
        
        if meas == 0:
            successes += 1
    
    runtime = time.time() - start
    return successes / trials, runtime

def benchmark(p_errors, trials):
    results = []
    for p in p_errors:
        success_rate, runtime = run_qiskit_simulation(p, trials)

        results.append({
            'p': p,
            'Success rate': success_rate,
            'Time': runtime,

        })

        print(f"p={p:.5f} | Success rate={success_rate:.4f} | Time={runtime:.3f}s")
    return results

def plot_results(results):
    ps = [r['p'] for r in results]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ps, [r['Success rate'] for r in results], 'o-', label='With ML Decoder')
    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate")
    plt.title("5-Qubit Code — ML Decoder Advantage")
    plt.legend()

    #plt.subplot(1, 2, 2)
    #plt.plot(ps, [r['no_dec_time'] for r in results], 'o-', label='No Decoder')
    #plt.plot(ps, [r['dec_time'] for r in results], 's--', label='With ML Decoder')
    #plt.xlabel("Physical error rate")
    #plt.ylabel("Runtime (s)")
    #plt.title("Runtime Performance")
    #plt.legend()

    plt.tight_layout()
    plt.savefig("qec_benchmark_ml.png")
    plt.show()
# Example usage
if __name__ == "__main__":
    p_errors = np.linspace(0.0005, 0.1, 50)  # Keep in correctable regime
    trials = 10000
    results = benchmark(p_errors, trials)
    plot_results(results)
