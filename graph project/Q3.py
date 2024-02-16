import itertools
import random

import networkx as nx
import matplotlib.pyplot as plt
import itertools
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
0.
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
def twisted_graph(num_nodes):

    graph = nx.Graph()

    graph.add_nodes_from(range(1, 2 * num_nodes + 1))

    for i in range(1, num_nodes + 1):
        graph.add_edge(i, (i % num_nodes) + 1)

    for i in range(1, num_nodes + 1):
        graph.add_edge(i, i + num_nodes)

    for i in range(num_nodes + 1, 2 * num_nodes + 1):
        graph.add_edge(i, ((i - num_nodes) % num_nodes) + num_nodes + 1)

    return graph
def tetrahedron_graph():

    G = nx.Graph()


    G.add_nodes_from([1, 2, 3, 4])
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(1, 4)
    G.add_edge(2, 3)
    G.add_edge(2, 4)
    G.add_edge(3, 4)

    return G
def cube_graph(n):

    G = nx.Graph()


    nodes = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]
    G.add_nodes_from(nodes)


    for i, j, k in nodes:
        if i < n - 1:
            G.add_edge((i, j, k), (i+1, j, k))
        if j < n - 1:
            G.add_edge((i, j, k), (i, j+1, k))
        if k < n - 1:
            G.add_edge((i, j, k), (i, j, k+1))

    return G
def octahedron():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5])
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)])
    return G
def icosahedron():
    G = nx.Graph()

    # Add nodes representing the vertices
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    # Add edges representing the connections between vertices
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                     (1, 6), (1, 7), (1, 8), (1, 9), (1, 10),
                     (2, 6), (2, 7), (2, 9), (2, 10), (2, 11),
                     (3, 7), (3, 8), (3, 10), (3, 11), (3, 6),
                     (4, 8), (4, 9), (4, 11), (4, 6), (4, 7),
                     (5, 9), (5, 10), (5, 11), (5, 6), (5, 8)])

    return G
#Hypercube Graph

graph = tetrahedron_graph()

# octahedron Graph
graph1 =octahedron()

# Cube Graph
graph2 = cube_graph(2)

# Icosahedron
graph3 = icosahedron()
#soccer ball
#:)))))))
def truncated_tetrahedron_graph():
    G = nx.Graph()
    # Add nodes
    vertices = [
        (1, 1, 1), (1, 3, 1), (1, 3, 3), (1, 1, 3),
        (3, 1, 1), (3, 3, 1), (3, 3, 3), (3, 1, 3),
        (2, 2, 0), (2, 0, 2), (0, 2, 2), (2, 4, 2)
    ]
    G.add_nodes_from(vertices)
    # Add edges
    edges = [
        (vertices[i], vertices[j])
        for i, j in [
            (0, 1), (0, 3), (0, 4), (1, 2), (1, 5),
            (2, 3), (2, 6), (3, 7), (4, 5), (4, 7),
            (5, 6), (6, 7), (0, 8), (1, 8), (2, 8),
            (3, 8), (4, 9), (5, 9), (8, 9), (4, 10),
            (5, 10), (8, 10), (6, 11), (7, 11), (8, 11)
        ]
    ]
    G.add_edges_from(edges)
    return G
graph4= truncated_tetrahedron_graph()

# Torus Graph
graph5=nx.grid_2d_graph(2, 2, periodic=True)

#Hypercube Graph

graph6= nx.generators.hypercube_graph(2)

#graph5 = nx.Graph()
# Add edges to the graph
#graph5.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (8, 5), (1, 5), (2, 6), (3, 7), (4, 8)])



# Draw the graph
#nx.draw(antiprism, with_labels=True, node_color='lightblue', node_size=800)
#plt.title("Your Graph")
#plt.show()
print("Nodes:"+str(count_nodes(graph)))
print("Edges:"+str(count_edges(graph)))
print("OC:"+str(oc_check(graph)))
print("Diameter:"+str(calculate_diameter(graph)))
print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(graph)))))
print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(graph)))))
print("-------------------------------------------------------------")
print("Nodes:"+str(count_nodes(graph1)))
print("Edges:"+str(count_edges(graph1)))
print("OC:"+str(oc_check(graph1)))
print("Diameter:"+str(calculate_diameter(graph1)))
print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(graph1)))))
print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(graph1)))))
print("-------------------------------------------------------------")
print("Nodes:"+str(count_nodes(graph2)))
print("Edges:"+str(count_edges(graph2)))
print("OC:"+str(oc_check(graph2)))
print("Diameter:"+str(calculate_diameter(graph2)))
print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(graph2)))))
print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(graph2)))))
print("-------------------------------------------------------------")
print("Nodes:"+str(count_nodes(graph3)))
print("Edges:"+str(count_edges(graph3)))
print("OC:"+str(oc_check(graph3)))
print("Diameter:"+str(calculate_diameter(graph3)))
#print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(graph3)))))
#print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(graph3)))))
print("-------------------------------------------------------------")
print("Nodes:"+str(12))
print("Edges:"+str(18))
print("OC:"+str(oc_check(graph4)))
print("Diameter:"+str(calculate_diameter(graph4)))
#print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(graph4)))))
#print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(graph4)))))
print("-------------------------------------------------------------")
print("Nodes:"+str(count_nodes(graph5)))
print("Edges:"+str(count_edges(graph5)))
print("OC:"+str(oc_check(graph5)))
print("Diameter:"+str(calculate_diameter(graph5)))
print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(graph5)))))
print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(graph5)))))
print("-------------------------------------------------------------")

print("Nodes:"+str(count_nodes(graph6)))
print("Edges:"+str(count_edges(graph6)))
print("OC:"+str(oc_check(graph6)))
print("Diameter:"+str(calculate_diameter(graph6)))
print("NS:"+str(ns_check( np.array(nx.to_numpy_matrix(graph6)))))
print("Symmetry:"+str(symmetry_check( np.array(nx.to_numpy_matrix(graph6)))))
print("-------------------------------------------------------------")