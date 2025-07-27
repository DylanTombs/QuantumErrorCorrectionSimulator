import stim

def build3QubitCodeCircuit():
    circuit = stim.Circuit()

    # Create |+⟩ state on qubit 0
    circuit.append_operation("H", [0])

    # Encode into repetition code: (|000⟩ + |111⟩)/√2
    circuit.append_operation("CNOT", [0, 1])
    circuit.append_operation("CNOT", [0, 2])

    return circuit

# Build circuit
circuit = build3QubitCodeCircuit()
print("Stim Circuit:")
print(circuit)

# Simulate with TableauSimulator
sim = stim.TableauSimulator()
sim.do_circuit(circuit)

# Get stabilizers of the state
stabilizers = sim.current_inverse_tableau().to_stabilizers()

print("\nStabilizers of the encoded state:")
for stab in stabilizers:
    print(stab)

