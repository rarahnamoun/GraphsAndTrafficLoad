import itertools

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

def oc_check(graph):
    node_cut = nx.minimum_node_cut(graph)
    min_edge_cut_value = nx.minimum_edge_cut(graph)
    min_degree = min(dict(graph.degree()).values())
    print( node_cut,min_edge_cut_value,min_degree)
    return  len(node_cut)==len(min_edge_cut_value) == min_degree




def edge_set(adj_matrix):
    edge_set = []
    for row in range(len(adj_matrix)):
        for col in range(len(adj_matrix[0])):
            if (list(np.sort([row, col])) not in edge_set):
                if adj_matrix[row][col] == 1:
                    edge_set.append(list(np.sort([row, col])))
    return sorted(edge_set)



def edge_permutation(vertex_perm, original_edge_set):
    edge_set_perm = []
    for edge in original_edge_set:
        edge_set_perm.append(edge.copy())
    for edge in edge_set_perm:
        edge[0] = vertex_perm[edge[0]]
        edge[1] = vertex_perm[edge[1]]
    return [original_edge_set, edge_set_perm]

def get_graph_order(adj_matrix):
    if len(adj_matrix) != len(adj_matrix[0]):
        return -1
    else:
        return len(adj_matrix)
def get_all_vertex_permutations(adj_matrix):
    if get_graph_order(adj_matrix) > 8:
        print("Stoppped")
        return -1
    all_adj_matrix = []
    original_edge_set = edge_set(adj_matrix)
    idx = list(range(len(adj_matrix)))
    possible_idx_combinations = [
        list(i) for i in itertools.permutations(idx, len(idx))
    ]
    for idx_comb in possible_idx_combinations:
        a = adj_matrix
        a = a[idx_comb]
        a = np.transpose(np.transpose(a)[idx_comb])
        all_adj_matrix.append({
            "perm_vertex":
                idx_comb,
            "perm_edge":
                edge_permutation(idx_comb, original_edge_set),
            "adj_matrix":
                a
        })

    return all_adj_matrix


def generate_automorphisms(adj_matrix):
  automorphisms_list = []
  for isograph in get_all_vertex_permutations(adj_matrix):
      if edge_set(adj_matrix) == edge_set(isograph["adj_matrix"]):
          automorphisms_list.append(isograph)
  return automorphisms_list

def ns_check(adj_matrix):
    vertex_comb = list(itertools.permutations(range(len(adj_matrix)), 2))
    automorphisms_perm = [
        auto["perm_vertex"] for auto in generate_automorphisms(adj_matrix)
    ]
    check = [None] * len(vertex_comb)
    for pair_vertex_idx in range(len(vertex_comb)):
        for auto_idx in range(len(automorphisms_perm)):
            if automorphisms_perm[auto_idx][vertex_comb[pair_vertex_idx][
                    0]] == vertex_comb[pair_vertex_idx][1]:
                check[pair_vertex_idx] = True
    return all(check)
def count_edges(graph):
    return graph.number_of_edges()
def count_nodes(graph):
    return graph.number_of_nodes()

def calculate_diameter(graph):
    return nx.diameter(graph)
def symmetry_check(adj_matrix):
    edge_comb = list(itertools.permutations(edge_set(adj_matrix), 2))
    automorphisms_perm = [
        auto["perm_edge"] for auto in generate_automorphisms(adj_matrix)
    ]
    check = [None] * len(edge_comb)
    for pair_edges_idx in range(len(edge_comb)):
      for auto_idx in range(len(automorphisms_perm)):
        edge_idx_to_check = automorphisms_perm[auto_idx][0].index(edge_comb[pair_edges_idx][0])
        if automorphisms_perm[auto_idx][1][edge_idx_to_check] == edge_comb[pair_edges_idx][1]:
          check[pair_edges_idx] = True
    return all(check)
def calculate_traffic_load(G):
    # Set the initial maximum traffic load to 0
    Lmax = 0

    # List to store the corresponding traffic loads
    traffic_loads = []

    # Simulate node destruction and calculate traffic load for different percentages
    num_iterations = 100
    for percent in range(0, 101, 10):  # Vary the percentage from 0 to 100 with a step of 10
        # Calculate the number of nodes to destroy based on the percentage
        num_nodes_to_destroy = int(percent / 100 * len(G))

        # Perform node destruction
        for i in range(num_nodes_to_destroy):
            # Check if the graph is empty
            if len(G) == 0:
                break

            # Select a random node to destroy
            node_to_destroy = random.choice(list(G.nodes()))

            # Remove the node and its associated edges
            G.remove_node(node_to_destroy)

        # Calculate shortest paths from source to destination
        shortest_paths = nx.shortest_path(G)

        # Calculate the traffic load (L) by equally distributing traffic flow on shortest paths
        L = 0
        for source in shortest_paths:
            for destination in shortest_paths[source]:
                if source < destination:  # Process each edge only once
                    path = shortest_paths[source][destination]
                    traffic_flow = -1 / len(path)  # Negative traffic flow
                    L += traffic_flow

        # Update the maximum traffic load (Lmax) if necessary
        if L < Lmax:
            Lmax = L

        # Store the traffic load value
        traffic_loads.append(L)

    # Plot the traffic load without showing the percentage of node destruction
    plt.style.use('rose-pine-moon')
    plt.plot(range(0, 101, 10), traffic_loads, marker='o')
    plt.title("Traffic Load vs. Percentage of Node Destruction")
    plt.xlabel("Percentage of Node Destruction")
    plt.ylabel("Traffic Load (L)")
    plt.grid(True)
    plt.show()

    # Print the maximum traffic load observed
    print(f"\nMaximum Traffic Load (Lmax): {-Lmax}")


G0=nx.complete_graph(4)

print("Nodes:"+str(count_nodes(G0)))
print("Edges:"+str(count_edges(G0)))
print("OC:"+str(oc_check(G0)))
print("Diameter:"+str(calculate_diameter(G0)))
print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(G0)))))
print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(G0)))))
calculate_traffic_load(G0)



# Define the size of the torus
rows = 2
columns = 2

G = nx.grid_2d_graph(rows, columns)

for i in range(rows):
    for j in range(columns):

        G.add_edge((i, j), (i, (j + 1) % columns))
        G.add_edge((i, j), ((i + 1) % rows, j))





print("Nodes:"+str(count_nodes(G)))
print("Edges:"+str(count_edges(G)))
print("OC:"+str(oc_check(G)))
print("Diameter:"+str(calculate_diameter(G)))
print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(G)))))
print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(G)))))
calculate_traffic_load(G)


G1=nx.cycle_graph(5)

print("Nodes:"+str(count_nodes(G1)))
print("Edges:"+str(count_edges(G1)))
print("OC:"+str(oc_check(G1)))
print("Diameter:"+str(calculate_diameter(G1)))
print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(G1)))))
print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(G1)))))
calculate_traffic_load(G1)



def prism_graph(num_nodes):
    graph = nx.Graph()

    graph.add_nodes_from(range(1, 2 * num_nodes + 1))

    for i in range(1, num_nodes + 1):
        graph.add_edge(i, (i % num_nodes) + 1)

    for i in range(1, num_nodes + 1):
        graph.add_edge(i, i + num_nodes)

    for i in range(num_nodes + 1, 2 * num_nodes + 1):
        graph.add_edge(i, ((i - num_nodes) % num_nodes) + num_nodes + 1)

    return graph
G2=prism_graph(3)
print("Nodes:"+str(count_nodes(G2)))
print("Edges:"+str(count_edges(G2)))
print("OC:"+str(oc_check(G2)))
print("Diameter:"+str(calculate_diameter(G2)))
print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(G2)))))
print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(G2)))))
calculate_traffic_load(G2)



def antiprism_graph(num_nodes):
    # Create a graph
    graph = nx.Graph()

    # Add nodes
    graph.add_nodes_from(range(1, num_nodes + 1))

    # Shuffle the nodes
    shuffled_nodes = list(graph.nodes)
    random.shuffle(shuffled_nodes)

    while shuffled_nodes:
        node = shuffled_nodes.pop()
        neighbors = list(graph.neighbors(node))

        # Collect the available neighbors
        available_neighbors = [n for n in shuffled_nodes if graph.degree[n] < 4]

        while graph.degree[node] < 4 and available_neighbors:
            neighbor = random.choice(available_neighbors)
            graph.add_edge(node, neighbor)
            neighbors.append(neighbor)
            available_neighbors.remove(neighbor)

    # Check if the graph has 2*nodes edges
    if graph.number_of_edges() == 2 * num_nodes:
        return graph
    else:
        return  antiprism_graph(num_nodes)

G3=antiprism_graph(6)
print("Nodes:"+str(count_nodes(G3)))
print("Edges:"+str(count_edges(G3)))
print("OC:"+str(oc_check(G3)))
print("Diameter:"+str(calculate_diameter(G3)))
print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(G3)))))
print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(G3)))))
calculate_traffic_load(G3)

def hamming_binary(chromosome_len):
    """Generate a binary Hamming Graph, where each genotype is composed by chromosome_len bases and each base can take only two values. H(chromosome_len, 2).

    steps to generate an Hamming graph:

     * create 2^chromosome_len nodes, each for a different binary string
     * for each node, find the connected nodes by flipping one position at a time.
    """
    space = nx.Graph()

    # create all nodes
    all_nodes = range(0, 2 ** chromosome_len)

    space.add_nodes_from(all_nodes)

    # for each node, find neighbors
    for node in space.nodes():
        [space.add_edge(node, mutate_node(node, base)) for base in range(chromosome_len)]
    return space


def mutate_node(node, n):
    """Generate a mutational neighbor of a node.

    Select the loci to be mutated by left-shifting a bit by n. Then do a bitwise
    XOR to do the mutation.

    Example:
    Node 26 =           11010
    n = 2: 00001 << 2 = 00100
                        -----
    XOR:                11110

    Example 2:
    Node 26 =           11010
    n = 1: 00001 << 1 = 00010
                        -----
    XOR:                11000
    """

    return node ^ (1 << n)
G4=hamming_binary(3)
print("Nodes:"+str(count_nodes(G4)))
print("Edges:"+str(count_edges(G4)))
print("OC:"+str(oc_check(G4)))
print("Diameter:"+str(calculate_diameter(G4)))
print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(G4)))))
print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(G4)))))
calculate_traffic_load(G4)