import time
import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator
from Circuits.Qubit5Qiskit import build_5qubit_qiskit_circuit, decode_qiskit_syndrome


def run_single_experiment(p_error: float, trials: int, seed: int = 0) -> dict:
    np.random.seed(seed)
    simulator = AerSimulator(seed_simulator=seed)
    
    start = time.perf_counter()
    successes = 0
    
    # Pre-build noisy circuit
    base_qc = build_5qubit_qiskit_circuit(p_error)
    
    for _ in range(trials):
        # Run encoded noisy circuit
        result = simulator.run(base_qc, shots=1).result()
        syndrome_bits = list(result.get_counts().keys())[0][::-1][:4]
        syndrome = [int(b) for b in syndrome_bits]
        
        x_corr, z_corr = decode_qiskit_syndrome(syndrome, p_error)
        
        # Correction circuit
        # (Could be prebuilt for speed)
        from qiskit import QuantumCircuit
        correction_qc = QuantumCircuit(9, 1)
        for q in range(5):
            if x_corr[q]:
                correction_qc.x(q)
            if z_corr[q]:
                correction_qc.z(q)
        
        correction_qc.h(0)
        for t in range(1, 5):
            correction_qc.cx(0, t)
        correction_qc.measure(0, 0)
        
        full_qc = base_qc.compose(correction_qc)
        meas = int(list(simulator.run(full_qc, shots=1).result().get_counts().keys())[0][0])
        
        successes += (meas == 0)
    
    runtime = time.perf_counter() - start
    return {
        "p": p_error,
        "success_rate": successes / trials,
        "runtime_s": runtime,
        "trials": trials,
        "seed": seed
    }


def benchmark(p_errors, trials, seed=0):
    results = [run_single_experiment(p, trials, seed) for p in p_errors]
    return pd.DataFrame(results)


if __name__ == "__main__":
    p_errors = np.linspace(0.0005, 0.1, 10)
    trials = 1000
    
    df = benchmark(p_errors, trials, seed=42)
    df.to_csv("single_benchmark_results.csv", index=False)
    
    print(df)
