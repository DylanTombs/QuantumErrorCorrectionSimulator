# fast_stim_5qubit.py
import stim, time, numpy as np

def build_5qubit_stim_circuit(p_error):
    c = stim.Circuit()
    # NOTE: preparing exact logical |0_L> by a compact encoder is tedious;
    # we can start from |0...0> and treat logical |0> = stabilizer projection.
    # For performance benchmarking we focus on syndrome + noise sampling.

    # Optionally encode (simple star-like mapping) - for now keep identity initial state
    # Apply depolarizing noise on data qubits 0..4
    for q in range(5):
        c.append("DEPOLARIZE1", [q], p_error)

    # Stabilizers S1..S4 using ancillas 5..8
    # S1 = X Z Z X I  (ancilla 5)
    c.append("H", [5])
    c.append("CNOT", [0,5])
    c.append("CZ", [1,5])
    c.append("CZ", [2,5])
    c.append("CNOT", [3,5])
    c.append("H", [5])
    c.append("M", [5])    # meas idx 0

    # S2 = I X Z Z X (ancilla 6)
    c.append("H", [6])
    c.append("CNOT", [1,6])
    c.append("CZ", [2,6])
    c.append("CZ", [3,6])
    c.append("CNOT", [4,6])
    c.append("H", [6])
    c.append("M", [6])    # meas idx 1

    # S3 = X I X Z Z (ancilla 7)
    c.append("H", [7])
    c.append("CNOT", [0,7])
    c.append("CZ", [2,7])
    c.append("CZ", [3,7])
    c.append("CNOT", [4,7])
    c.append("H", [7])
    c.append("M", [7])    # meas idx 2

    # S4 = Z X I X Z (ancilla 8)
    c.append("H", [8])
    c.append("CZ", [0,8])
    c.append("CNOT", [1,8])
    c.append("CNOT", [3,8])
    c.append("CZ", [4,8])
    c.append("H", [8])
    c.append("M", [8])    # meas idx 3

    # For logical measurement, measure parity of Z0..Z4 and mark as observable
    # measure data qubit 0 in Z basis and include as observable (for speed we choose one qubit)
    c.append("M", [0])
    c.append("OBSERVABLE_INCLUDE(1)", [0])

    return c

def run_5qubit_stim_bulk(p_error, trials=200_000, seed=123):
    circ = build_5qubit_stim_circuit(p_error)
    sampler = circ.compile_sampler()
    t0 = time.time()
    samples = sampler.sample(repetitions=trials, random_seed=seed)
    t1 = time.time()
    logical_obs = samples[:, -1]
    ler = logical_obs.mean()
    return ler, t1-t0

if __name__ == "__main__":
    p=0.02
    trials = 100_000
    ler, t = run_5qubit_stim_bulk(p, trials)
    print("Stim 5-qubit:", "p=",p,"trials=",trials,"LER=",ler,"time(s)=",t)
