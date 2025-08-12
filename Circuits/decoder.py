import math
import networkx as nx


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
    stabiliser_edges = [("X1", "X3", -math.log(p_error**2))]
    for u, v, w in stabiliser_edges:
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