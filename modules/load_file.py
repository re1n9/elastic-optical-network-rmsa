import networkx as nx
import numpy as np

# Load topology from file
def load_topology(file_path):
    G = nx.Graph() #create an empty graph
    with open(file_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            data = line.strip().split()
            node_a, node_b, length = int(data[3]), int(data[4]), float(data[5])
            G.add_edge(node_a, node_b, weight = length)
    return G

# Load traffic matrix from file
def load_traffic_matrix(file_path):
    matrix = np.loadtxt(file_path)
    traffic_requests = []
    num_codes = matrix.shape[0]

    for i in range(num_codes):
        for j in range(num_codes):
            if matrix[i][j] > 0:
                traffic_requests.append((i+1, j+1, matrix[i][j] * 10))
    return traffic_requests