import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

############################
# DECODER IMPLEMENTATION
############################

# Stabilizer generators for the 5-qubit code (Pauli strings)
STABILIZERS = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ"
]

PAULIS = ['I', 'X', 'Y', 'Z']

def pauli_commutes(p1, p2):
    """Return True if two Pauli strings commute (ignoring global phase)."""
    anti_count = 0
    for a, b in zip(p1, p2):
        if a == 'I' or b == 'I':
            continue
        if a != b:
            anti_count += 1
    return (anti_count % 2) == 0

def syndrome_for_error(error_str):
    """Compute 4-bit syndrome for given Pauli error string using STABILIZERS."""
    return tuple(0 if pauli_commutes(error_str, stab) else 1 for stab in STABILIZERS)

def build_decoder_table():
    """
    Build syndrome -> minimal single-qubit correction table.
    We try all single-qubit X/Y/Z errors and record the syndrome mapping.
    (This is optimal for correcting single-qubit errors under depolarizing noise.)
    """
    table = {}
    # single-qubit errors
    for q in range(5):
        for p in ['X', 'Y', 'Z']:
            err = ['I'] * 5
            err[q] = p
            err_str = "".join(err)
            syn = syndrome_for_error(err_str)
            # If multiple single-qubit errors map to same syndrome, prefer first found (all are weight-1)
            if syn not in table:
                table[syn] = err_str
    # identity (no error) mapping
    table[(0,0,0,0)] = "IIIII"
    return table

DECODER_TABLE = build_decoder_table()

def decode_5qubit_syndrome(syndrome_bits, p_error=None):
    """
    Lookup-based decoder: returns list of 5 tuples (x_bit, z_bit),
    where (1,0)=X, (0,1)=Z, (1,1)=Y, (0,0)=I.
    """
    syn = tuple(int(b) for b in syndrome_bits)
    corr_str = DECODER_TABLE.get(syn, "IIIII")
    corrections = []
    for c in corr_str:
        if c == 'I':
            corrections.append((0,0))
        elif c == 'X':
            corrections.append((1,0))
        elif c == 'Z':
            corrections.append((0,1))
        elif c == 'Y':
            corrections.append((1,1))
    return corrections

############################
# QISKIT IMPLEMENTATIONS
############################

def build_5qubit_qiskit():
    """
    Build 5-qubit code circuit:
      - data qubits: 0..4
      - stabilizer ancillas: 5..8 (measured to classical bits 0..3)
      - logical-Z ancilla: 9 (measured to classical bit 4)
      - logical-X ancilla: 10 (measured to classical bit 5)
    Classical bits layout (index):
      0..3 : syndrome S1..S4
      4    : logical Z parity (Z_L)
      5    : logical X parity (X_L)
    """
    qc = QuantumCircuit(11, 6)  # 5 data + 4 stabilizer ancilla + 2 logical ancilla, 6 classical bits

    # --- Encode logical |0> into the 5-qubit code (simple circuit) ---
    # This is a common simple encoder choice for the 5-qubit code (entangle qubit 0 into others)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(0, 4)

    # --- Stabilizer measurement circuits ---
    # S1 = X Z Z X I  -> ancilla 5 -> cbit 0
    qc.h(5)
    qc.cx(0, 5)
    qc.cz(1, 5)
    qc.cz(2, 5)
    qc.cx(3, 5)
    qc.h(5)
    qc.measure(5, 0)

    # S2 = I X Z Z X  -> ancilla 6 -> cbit 1
    qc.h(6)
    qc.cx(1, 6)
    qc.cz(2, 6)
    qc.cz(3, 6)
    qc.cx(4, 6)
    qc.h(6)
    qc.measure(6, 1)

    # S3 = X I X Z Z  -> ancilla 7 -> cbit 2
    qc.h(7)
    qc.cx(0, 7)
    qc.cz(2, 7)
    qc.cz(3, 7)
    qc.cx(4, 7)
    qc.h(7)
    qc.measure(7, 2)

    # S4 = Z X I X Z  -> ancilla 8 -> cbit 3
    qc.h(8)
    qc.cz(0, 8)
    qc.cx(1, 8)
    qc.cx(3, 8)
    qc.cz(4, 8)
    qc.h(8)
    qc.measure(8, 3)

    # --- Logical Z parity measurement onto ancilla 9 ---
    # Z_L = Z0 Z1 Z2 Z3 Z4
    # Use ancilla 9 prepared in |0>, CNOT from each data -> ancilla9,
    # then measure ancilla9 to get parity (even -> 0, odd -> 1)
    for q in range(5):
        qc.cx(q, 9)
    qc.measure(9, 4)

    # --- Logical X parity measurement onto ancilla 10 ---
    # X_L = X0 X1 X2 X3 X4
    # Convert data X's to Z's by H on data, do CNOTs data->ancilla10, H back.
    for q in range(5):
        qc.h(q)
    for q in range(5):
        qc.cx(q, 10)
    for q in range(5):
        qc.h(q)
    qc.measure(10, 5)

    return qc

def noise_model_qiskit(p_error):
    """Depolarizing noise model applied to 1- and 2-qubit gates."""
    noise = NoiseModel()
    err1 = depolarizing_error(p_error, 1)
    err2 = depolarizing_error(p_error, 2)
    noise.add_all_qubit_quantum_error(err1, ['h', 'x', 'y', 'z', 'reset'])
    noise.add_all_qubit_quantum_error(err2, ['cx', 'cz'])
    return noise

############################
# RUNNERS
############################

def parse_counts_key(bits_str):
    """
    Convert Qiskit's returned bitstring to a list of ints in our classical-bit order:
    We use bits_list = [int(b) for b in bits_str[::-1]] so that index 0 corresponds to classical bit 0.
    """
    return [int(b) for b in bits_str[::-1]]

def run_with_decoder(p_error, trials):
    backend = AerSimulator()
    qc = build_5qubit_qiskit()
    noise = noise_model_qiskit(p_error)

    # transpile and run
    t0 = time.time()
    job = backend.run(transpile(qc, backend), shots=trials, noise_model=noise)
    result = job.result()
    t1 = time.time()

    error_count = 0
    # counts keys are bitstrings like '01011' (big-endian). We reverse to get our mapping.
    for bits, count in result.get_counts().items():
        bits_list = parse_counts_key(bits)  # index 0..5: c0..c5
        syndrome = bits_list[0:4]          # S1..S4
        logical_z_bit = bits_list[4]       # Z_L parity
        logical_x_bit = bits_list[5]       # X_L parity

        # Decoder returns per-qubit (x,z) correction bits
        corrections = decode_5qubit_syndrome(syndrome, p_error)
        x_corrections = [c[0] for c in corrections]
        z_corrections = [c[1] for c in corrections]

        # Applying corrections flips parity if total parity of corrections is odd
        total_x = sum(x_corrections) % 2
        total_z = sum(z_corrections) % 2

        corrected_logical_z = logical_z_bit ^ total_z
        corrected_logical_x = logical_x_bit ^ total_x

        # Logical error if either corrected logical Z or X is non-zero (expected logical |0> => Z=0 and X=0)
        if corrected_logical_z != 0 or corrected_logical_x != 0:
            error_count += count

    ler = error_count / trials
    return ler, t1 - t0

def run_without_decoder(p_error, trials):
    backend = AerSimulator()
    qc = build_5qubit_qiskit()
    noise = noise_model_qiskit(p_error)

    t0 = time.time()
    job = backend.run(transpile(qc, backend), shots=trials, noise_model=noise)
    result = job.result()
    t1 = time.time()

    error_count = 0
    for bits, count in result.get_counts().items():
        bits_list = parse_counts_key(bits)
        logical_z_bit = bits_list[4]
        logical_x_bit = bits_list[5]
        if logical_z_bit != 0 or logical_x_bit != 0:
            error_count += count

    ler = error_count / trials
    return ler, t1 - t0

############################
# BENCHMARKING & PLOTTING
############################

def benchmark(p_errors: list[float], trials: int):
    results = []
    for p in p_errors:
        ler_no_dec, time_no_dec = run_without_decoder(p, trials)
        ler_dec, time_dec = run_with_decoder(p, trials)

        results.append({
            'p': p,
            'no_dec_ler': ler_no_dec,
            'no_dec_time': time_no_dec,
            'dec_ler': ler_dec,
            'dec_time': time_dec
        })

        print(f"p={p:.3f} | "
              f"No Decoder: LER={ler_no_dec:.4f}, Time={time_no_dec:.3f}s | "
              f"With Decoder: LER={ler_dec:.4f}, Time={time_dec:.3f}s")

    return results

def plot_results(results):
    ps = [r['p'] for r in results]

    plt.figure(figsize=(12, 5))

    # LER plot
    plt.subplot(1, 2, 1)
    plt.plot(ps, [r['no_dec_ler'] for r in results], 'o-', label='No Decoder')
    plt.plot(ps, [r['dec_ler'] for r in results], 's--', label='With Decoder')
    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate")
    plt.title("Error Correction Performance (5-Qubit Code)")
    plt.legend()

    # Runtime plot
    plt.subplot(1, 2, 2)
    plt.plot(ps, [r['no_dec_time'] for r in results], 'o-', label='No Decoder')
    plt.plot(ps, [r['dec_time'] for r in results], 's--', label='With Decoder')
    plt.xlabel("Physical error rate")
    plt.ylabel("Runtime (s)")
    plt.title("Computational Performance")
    plt.legend()

    plt.tight_layout()
    plt.savefig("qec_benchmark.png")
    plt.show()

############################
# MAIN
############################

if __name__ == "__main__":
    # choose p range and trials
    p_errors = np.linspace(0.01, 0.15, 5)
    trials = 3000  # increase if you have time; 3000 gives reasonable statistics

    print(f"Running benchmark with {trials} trials per p...")
    results = benchmark(p_errors, trials)
    plot_results(results)

