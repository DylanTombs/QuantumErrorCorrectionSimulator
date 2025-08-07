import stim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

def simulate_3qubit_code_trial(p_error: float):
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
    """
    Given a 2-bit syndrome (from ZZI and IZZ), return which qubit to correct
    using MWPM decoding logic (simplified for 3-qubit repetition code).
    """
    # Syndrome bits: (s01, s12)
    s01, s12 = syndrome

    # Map syndrome to possible detection events
    detection_events = []
    if s01:
        detection_events.append("s01")
    if s12:
        detection_events.append("s12")

    # If no detection events, return no correction
    if not detection_events:
        return [0, 0, 0]

    # Build decoder graph
    G = nx.Graph()
    for i in range(len(detection_events)):
        for j in range(i + 1, len(detection_events)):
            u, v = detection_events[i], detection_events[j]
            # Assign weight = distance (simplified to 1 here)
            G.add_edge(u, v, weight=1)

    # Match detection events in pairs (min-weight perfect matching)
    matching = nx.algorithms.matching.min_weight_matching(G, maxcardinality=True)

    # Simple decoding logic: decide which qubit to flip based on detection events
    if syndrome == (1, 0):  # error between qubit 0 and 1 → flip qubit 0
        return [1, 0, 0]
    elif syndrome == (1, 1):  # center syndrome → flip middle qubit
        return [0, 1, 0]
    elif syndrome == (0, 1):  # error between qubit 1 and 2 → flip qubit 2
        return [0, 0, 1]
    else:
        return [0, 0, 0]

def run_experiment(p_error: float, trials: int = 100_000):
    successes = 0
    for i in range(trials):
        if simulate_3qubit_code_trial(p_error):
            successes += 1
        if i % 1000 == 0:  # Print every 1000 trials
            print(f"Completed {i}/{trials} trials (p={p_error})")
    return successes / trials

if __name__ == "__main__":
    #p_values = [0.01, 0.05, 0.1, 0.15, 0.2]
    p_values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]  # More detail where it matters
    results = {p: run_experiment(p, 10_000) for p in p_values}

    # Plotting
    plt.figure()
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel("Physical Bit-Flip Probability (p)")
    plt.ylabel("Logical Error Rate")
    plt.title("3-Qubit Bit-Flip Code: Logical vs Physical Error Rate")
    plt.grid(True)
    plt.show()
