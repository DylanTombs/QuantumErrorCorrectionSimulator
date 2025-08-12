import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Any

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import stim
from Circuits.Qubit3Qiskit import build_3qubit_qiskit_circuit, decode_3qubit_syndrome
from Circuits.Qubit5Qiskit import build_5qubit_qiskit_circuit, decode_qiskit_syndrome
from Circuits.Quibit3Stim import build_3qubit_stim_circuit, decode_3qubit_syndrome_stim   
from Circuits.Qubit5Stim import build_5qubit_stim_circuit, decode_5qubit_syndrome_stim
    

def run_qiskit_5qubit(p_error: float, trials: int, seed: int) -> Dict[str, Any]:
    """Run 5-qubit code simulation in Qiskit with fresh noise per trial."""

    np.random.seed(seed)
    simulator = AerSimulator()
    successes = 0

    start = time.perf_counter()
    for _ in range(trials):
        qc = build_5qubit_qiskit_circuit(p_error)

        # Syndrome extraction
        result = simulator.run(qc, shots=1).result()
        syndrome_bits = list(result.get_counts().keys())[0][::-1][:4]
        syndrome = [int(b) for b in syndrome_bits]

        # Decode
        x_corr, z_corr = decode_qiskit_syndrome(syndrome, p_error)

        # Apply corrections and measure logical
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

        full_qc = qc.compose(correction_qc)
        meas = int(list(simulator.run(full_qc, shots=1).result().get_counts().keys())[0][0])
        successes += (meas == 0)

    runtime = time.perf_counter() - start
    return {"success_rate": successes / trials, "runtime_s": runtime}


def run_qiskit_3qubit(p_error: float, trials: int, seed: int) -> Dict[str, Any]:
    """Run 3-qubit repetition code simulation in Qiskit with error correction."""
    
    np.random.seed(seed)
    simulator = AerSimulator()
    successes = 0
    
    start = time.perf_counter()
    for _ in range(trials):
        # Build circuit with noise and syndrome measurement
        qc = build_3qubit_qiskit_circuit(p_error)
        
        # Extract syndrome
        result = simulator.run(qc, shots=1).result()
        syndrome_bits = list(result.get_counts().keys())[0][::-1][:2]  # Get syndrome bits
        syndrome = [int(b) for b in syndrome_bits]
        
        # Decode and apply corrections
        corrections = decode_3qubit_syndrome(syndrome)
        
        # Apply corrections and measure logical qubit
        correction_qc = QuantumCircuit(5, 1)
        for q in range(3):
            if corrections[q]:
                correction_qc.x(q)  # Apply X correction
        
        # Measure logical qubit (majority vote or just qubit 0 for repetition code)
        correction_qc.measure(0, 0)
        
        # Combine circuits
        full_qc = qc.compose(correction_qc)
        
        # Run final measurement
        final_result = simulator.run(full_qc, shots=1).result()
        meas = int(list(final_result.get_counts().keys())[0][0])
        successes += (meas == 0)  # Count as success if logical |0> is preserved
    
    runtime = time.perf_counter() - start
    return {"success_rate": successes / trials, "runtime_s": runtime}

def run_stim_3qubit(p_error: float, trials: int, seed: int) -> Dict[str, Any]:
    """Run 3-qubit repetition code simulation in Stim with error correction."""
    np.random.seed(seed)
    successes = 0
    
    start = time.perf_counter()
    for _ in range(trials):
        # Build circuit with noise and syndrome measurement
        circuit = build_3qubit_stim_circuit(p_error)
        
        # Run syndrome measurement
        sampler = circuit.compile_sampler()
        syndrome_sample = sampler.sample(1)[0]  # Get single sample
        syndrome = syndrome_sample.tolist()
        
        # Decode and apply corrections
        corrections = decode_3qubit_syndrome_stim(syndrome)
        
        # Create correction circuit
        correction_circuit = stim.Circuit()
        for q in range(3):
            if corrections[q]:
                correction_circuit.append("X", [q])
        
        # Measure logical qubit (qubit 0)
        correction_circuit.append("M", [0])
        
        # Combine circuits
        full_circuit = circuit + correction_circuit
        
        # Run final measurement
        final_sampler = full_circuit.compile_sampler()
        final_result = final_sampler.sample(1)[0]
        logical_measurement = final_result[-1]  # Last measurement is the logical readout
        
        successes += (logical_measurement == 0)
    
    runtime = time.perf_counter() - start
    return {"success_rate": successes / trials, "runtime_s": runtime}


def run_stim_5qubit(p_error: float, trials: int, seed: int) -> Dict[str, Any]:
    """Run 5-qubit code simulation in Stim with error correction."""
    np.random.seed(seed)
    successes = 0
    
    start = time.perf_counter()
    for _ in range(trials):
        # Build circuit with noise and syndrome measurement
        circuit = build_5qubit_stim_circuit(p_error)
        
        # Run syndrome measurement
        sampler = circuit.compile_sampler()
        syndrome_sample = sampler.sample(1)[0]
        syndrome = syndrome_sample.tolist()
        
        # Decode
        x_corr, z_corr = decode_5qubit_syndrome_stim(syndrome, p_error)
        
        # Apply corrections and measure logical
        correction_circuit = stim.Circuit()
        for q in range(5):
            if x_corr[q]:
                correction_circuit.append("X", [q])
            if z_corr[q]:
                correction_circuit.append("Z", [q])
        
        # Measure logical qubit (decode by undoing encoding)
        correction_circuit.append("H", [0])
        for t in range(1, 5):
            correction_circuit.append("CX", [0, t])
        correction_circuit.append("M", [0])
        
        # Combine circuits
        full_circuit = circuit + correction_circuit
        
        # Run final measurement
        final_sampler = full_circuit.compile_sampler()
        final_result = final_sampler.sample(1)[0]
        logical_measurement = final_result[-1]  # Last measurement is logical readout
        
        successes += (logical_measurement == 0)
    
    runtime = time.perf_counter() - start
    return {"success_rate": successes / trials, "runtime_s": runtime}

# ==== BENCHMARKING ENGINE ====
def benchmark_models(models: Dict[str, Callable], p_errors, trials, seeds):
    """Runs each model for each p and seed. Returns aggregated DataFrame."""
    rows = []
    for model_name, runner in models.items():
        for p in p_errors:
            for seed in seeds:
                result = runner(p, trials, seed)
                rows.append({
                    "model": model_name,
                    "p": p,
                    "trials": trials,
                    "seed": seed,
                    **result
                })
                print(f"[{model_name}] p={p:.5f}, seed={seed} | "
                      f"SR={result['success_rate']:.4f} | t={result['runtime_s']:.3f}s")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p_errors = np.linspace(0.0005, 0.1, 10)
    trials = 2_000  # per seed
    seeds = [42, 123, 999]

    models = {
        "qiskit_3bit": run_qiskit_3qubit,
        "stim_5qubit": run_stim_5qubit,
        "qiskit_5qubit": run_qiskit_5qubit,
        "qiskit_3bit": run_qiskit_3qubit,
    }

    df = benchmark_models(models, p_errors, trials, seeds)
    out_path = Path("benchmark_multi.csv")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved benchmark results to {out_path}")
