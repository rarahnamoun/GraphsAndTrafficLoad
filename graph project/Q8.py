import random
import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('rose-pine-moon')
def calculate_diameter(graph):
    return nx.diameter(graph)
def count_edges(graph):
    return graph.number_of_edges()
def count_nodes(graph):
    return graph.number_of_nodes()
def oc_check(graph):
    node_cut = nx.minimum_node_cut(graph)
    min_edge_cut_value = nx.minimum_edge_cut(graph)
    min_degree = min(dict(graph.degree()).values())
    print( node_cut,min_edge_cut_value,min_degree)
    return  len(node_cut)==len(min_edge_cut_value) == min_degree
def small_world_generate_two_level_network(base_network, hub_network, k, p):
    # Create a copy of the hub network with relabeled nodes
    relabeled_hub_network = nx.relabel_nodes(hub_network, {n: f'a{n}' for n in hub_network.nodes()})

    # Create an empty graph
    G = nx.Graph()

    # Add nodes from the base and relabeled hub networks
    G.add_nodes_from(base_network.nodes())
    G.add_nodes_from(relabeled_hub_network.nodes())

    # Add edges within the base and relabeled hub networks
    G.add_edges_from(base_network.edges())
    G.add_edges_from(relabeled_hub_network.edges())


    base_network_nodes = list(base_network.nodes())
    hub_network_nodes = list(relabeled_hub_network.nodes())

    for i in range(len(base_network_nodes)):
        # Select the two nearest neighbors of the base node
        base_node = base_network_nodes[i]
        base_neighbors = list(base_network.neighbors(base_node))
        if len(base_neighbors) < k:
            continue
        base_neighbors.sort(key=lambda x: base_network.degree(x), reverse=True)
        neighbor1, neighbor2 = base_neighbors[:2]

        # Select the two nearest neighbors of the hub node
        hub_node = hub_network_nodes[i]
        hub_neighbors = list(relabeled_hub_network.neighbors(hub_node))
        if len(hub_neighbors) < k:
            continue
        hub_neighbors.sort(key=lambda x: relabeled_hub_network.degree(x), reverse=True)
        neighbor3, neighbor4 = hub_neighbors[:2]

        # Connect the base node to the hub node and the two nearest hub neighbors
        G.add_edge(base_node, hub_node)
        G.add_edge(hub_node, neighbor3)
        G.add_edge(hub_node, neighbor4)

        # Rewire the connections to the base node's neighbors with probability p
        for neighbor in base_neighbors:
            if neighbor == neighbor1 or neighbor == neighbor2:
                continue
            if random.random() < p:
                G.remove_edge(base_node, neighbor)
                hub_neighbor = nx.utils.arbitrary_element(hub_network_nodes[:-1])
                G.add_edge(base_node, hub_neighbor)
                G.add_edge(hub_neighbor, neighbor)

    return G
def generate_semi_ba_two_level_network(base_network, hub_network, m):
    # Create a copy of the hub network with relabeled nodes
    relabeled_hub_network = nx.relabel_nodes(hub_network, {n: f'a{n}' for n in hub_network.nodes()})

    # Create an empty graph
    G = nx.Graph()

    # Add nodes from the base and relabeled hub networks
    G.add_nodes_from(base_network.nodes())
    G.add_nodes_from(relabeled_hub_network.nodes())

    # Add edges within the base and relabeled hub networks
    G.add_edges_from(base_network.edges())
    G.add_edges_from(relabeled_hub_network.edges())

    for i in range(m):

        base_node = nx.utils.arbitrary_element(base_network.nodes())
        hub_node = nx.utils.arbitrary_element(relabeled_hub_network.nodes())
        G.add_edge(base_node, hub_node)
        base_network.add_edge(base_node, hub_node)
        relabeled_hub_network.add_edge(hub_node, base_node)

    return G
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
def generate_two_level_network(base_network, hub_network, p_out):
    # Create a copy of the hub network with relabeled nodes
    relabeled_hub_network = nx.relabel_nodes(hub_network, {n: f'a{n}' for n in hub_network.nodes()})

    # Create an empty graph
    G = nx.Graph()

    # Add nodes from the base and relabeled hub networks
    G.add_nodes_from(base_network.nodes())
    G.add_nodes_from(relabeled_hub_network.nodes())

    # Add edges within the base and relabeled hub networks
    G.add_edges_from(base_network.edges())
    G.add_edges_from(relabeled_hub_network.edges())

    # Add edges between the base and relabeled hub networks
    for i in base_network.nodes():
        for j in relabeled_hub_network.nodes():
            # With probability p_out, connect nodes from different networks
            if random.random() < p_out:
                G.add_edge(i, j)

    return G

# Scenario1
base_network  =prism_graph(1000)
hub_network = nx.barabasi_albert_graph(4, 2)     # Example hub network
p_out = 2                     # Probability of an edge between the networks

network = generate_semi_ba_two_level_network(base_network, hub_network, p_out)
pos = nx.spring_layout(network)
#nx.draw(network, pos, with_labels=True)
#plt.show()
# Scenario 4-3
print("Nodes:"+str(count_nodes(network)))
print("Edges:"+str(count_edges(network)))
#print("OC:"+str(oc_check(network)))
print("Diameter:"+str(calculate_diameter(network)))
print("Node connect:"+str(nx.node_connectivity(network)))
print("Edge connect:"+str(nx.edge_connectivity(network)))
# Scenario2
base_network  =prism_graph(1000)
hub_network  = nx.watts_strogatz_graph(5, 2, 0)     # Example hub network
p_out = 2                     # Probability of an edge between the networks

network = generate_semi_ba_two_level_network(base_network, hub_network, p_out)
pos = nx.spring_layout(network)



#nx.draw(network, pos, with_labels=True)
#plt.show()# Scenario 4-3
print("Nodes:"+str(count_nodes(network)))
print("Edges:"+str(count_edges(network)))
#print("OC:"+str(oc_check(network)))
print("Diameter:"+str(calculate_diameter(network)))
print("Node connect:"+str(nx.node_connectivity(network)))
print("Edge connect:"+str(nx.edge_connectivity(network)))