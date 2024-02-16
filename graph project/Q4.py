import itertools
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import curve_fit

def find_set(sets, node):
    # Find the set containing the given node
    for s in sets:
        if node in s:
            return s

def contract_edge(graph, u, v):
    # Contract the edge (u,v) to a single node
    for w in graph[v]:
        if w != u:
            graph[u].append(w)
            graph[w].append(u)
        graph[w].remove(v)
    del graph[v]

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
        print("Stopped")
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
            "perm_vertex": idx_comb,
            "perm_edge": edge_permutation(idx_comb, original_edge_set),
            "adj_matrix": a
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
            if automorphisms_perm[auto_idx][vertex_comb[pair_vertex_idx][0]] == vertex_comb[pair_vertex_idx][1]:
                check[pair_vertex_idx] = True
    return all(check)

def oc_check(graph,n,p):
    node_cut = nx.minimum_node_cut(graph)
    min_edge_cut_value =nx.minimum_edge_cut(graph)
    min_degree = min(dict(graph.degree()).values())
    return len(node_cut) == len(min_edge_cut_value )== min_degree

def func_fit(x, a, b, c):
    return a * np.exp(-b * x) + c

results = []

# OC feature
n_vals = range(7,17)
p_vals = [2*np.sqrt(n)/n for n in n_vals]
num_graphs = 1000

for n in n_vals:
    num_passed = 0
    for i in range(num_graphs):
        d = np.sqrt(n)
        p = 2 * d / (n - 1)
        G = nx.gnp_random_graph(n, p)
        if(nx.is_connected(G)):
            if oc_check(G,n,p):
                num_passed += 1
        else:
            num_passed = num_passed
    percentage_passed = num_passed / num_graphs
    results.append(percentage_passed)

xdata = np.array(n_vals)
ydata = np.array(results)
popt, _ = curve_fit(func_fit, xdata, ydata)
plt.style.use('rose-pine-moon')
plt.plot(n_vals, results, 'o-', label='data')
plt.plot(xdata, func_fit(xdata, *popt), '-', label='fit')
plt.xlabel('Number of nodes')
plt.ylabel('Percentage of graphs passing oc_check')
plt.legend()
plt.show()