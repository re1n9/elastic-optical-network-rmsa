import itertools
import networkx as nx

# Generate k-shortest paths
def k_shortest_paths(graph, source, target, k=5):
    return list(itertools.islice(nx.shortest_simple_paths(graph, source, target, weight='weight'), k))