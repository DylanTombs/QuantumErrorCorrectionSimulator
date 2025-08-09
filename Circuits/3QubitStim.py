# fast_stim_qec.py
import stim
import numpy as np
import time
from tqdm import tqdm

def build_3qubit_stim_circuit(p_error, use_depolarize=False):
    c = stim.Circuit()
    # encode (|0_L> -> |000>)
    c.append("H", [0])
    c.append("CNOT", [0, 1])
    c.append("CNOT", [0, 2])

    # noise: let stim sample errors internally
    if use_depolarize:
        c.append("DEPOLARIZE1", [0, 1, 2], p_error)
    else:
        for q in [0,1,2]:
            c.append("X_ERROR", [q], p_error)

    # measure stabilizers with ancillas 3 and 4
    c.append("CNOT", [0, 3])
    c.append("CNOT", [1, 3])
    c.append("M", [3])  # s0
    c.append("CNOT", [1, 4])
    c.append("CNOT", [2, 4])
    c.append("M", [4])  # s1

    # Add an observable that indicates logical Z (parity of data qubits)
    # We can measure logical Z (Z0 Z1 Z2) by toggling an ancilla or directly use OBSERVABLE_INCLUDE
    # Stim supports OBSERVABLE_INCLUDE which accumulates parity of specific measurement results.
    # Here we'll construct parity of data Z measurements by adding a measure of each data qubit in Z.
    # Simple: measure Z on data qubit 0 at the end to decide final logical state (since repetition code).
    c.append("M", [0])
    # Make the last measurement index an observable for easy extraction:
    c.append("OBSERVABLE_INCLUDE(1)", [0])  # Add observable 0 = final Z result

    return c

def run_stim_bulk(circuit: stim.Circuit, trials: int, seed=None):
    sampler = circuit.compile_sampler()
    # sample returns an ndarray shape (trials, num_measurements)
    t0 = time.time()
    samples = sampler.sample(repetitions=trials, random_seed=seed)
    t1 = time.time()
    # last column was the observable we added (OBSERVABLE_INCLUDE makes it accessible via metadata, but to keep simple
    # we can index by -1 because we appended the final M last).
    logical_obs = samples[:, -1]  # 0/1 array
    logical_error_rate = logical_obs.mean()  # fraction '1'
    return logical_error_rate, t1 - t0

if __name__ == "__main__":
    trials = 200_000
    p = 0.02
    circ = build_3qubit_stim_circuit(p_error=p, use_depolarize=False)
    print("Stim circuit built. Num measurements:", circ.num_measurements)
    ler, runtime = run_stim_bulk(circ, trials, seed=12345)
    print(f"Stim: trials={trials}, p={p}, logical error rate={ler:.6f}, time={runtime:.3f}s")
