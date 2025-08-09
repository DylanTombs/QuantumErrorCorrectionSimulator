# qiskit_benchmark.py
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from  qiskit_aer.noise import NoiseModel, depolarizing_error

def build_3qubit_qiskit_circuit(p_error):
    qc = QuantumCircuit(3, 1)  # we'll measure one classical bit as logical readout
    # encode
    qc.h(0)
    qc.cx(0,1)
    qc.cx(0,2)
    # measure stabilizers with ancillas? for speed, we'll just measure code output (not exact mapping)
    # For fair comparison you should replicate exactly the Stim circuit including ancillas and measurement bits
    # For this demo we measure qubit 0 as logical readout
    qc.measure_all()
    return qc

def build_noise_model(p_error):
    noise = NoiseModel()
    err = depolarizing_error(p_error, 1)
    for q in range(3):
        noise.add_all_qubit_quantum_error(err, ['id', 'x', 'y', 'z', 'cx', 'cz', 'h'])
    return noise

def run_qiskit_bulk(p_error, trials):
    qc = build_3qubit_qiskit_circuit(p_error)
    backend = AerSimulator()
    noise = build_noise_model(p_error)
    tcirc = transpile(qc, backend)
    t0 = time.time()
    job = backend.run(tcirc, shots=trials, noise_model=noise)
    result = job.result()
    t1 = time.time()
    counts = result.get_counts()
    # compute logical error rate from counts: here measuring parity as demonstration
    total_shots = trials
    # naive: treat counts with bitstring where lsb corresponds to qubit 0 (depends on bit ordering)
    # compute fraction where logical readout is '1' (depends on your encoding)
    ones = 0
    for bitstr, cnt in counts.items():
        # adapt indexing as needed; for example take leftmost bit as qubit0:
        if bitstr[0] == '1':
            ones += cnt
    ler = ones / total_shots
    return ler, (t1-t0)

if __name__ == "__main__":
    p = 0.02
    trials = 100_000
    ler, runtime = run_qiskit_bulk(p, trials)
    print("Qiskit 3-qubit:", "p=",p,"trials=",trials,"LER=",ler,"time(s)=",runtime)
