import networkx as nx
import random
import matplotlib.pyplot as plt
def startswith(s, prefix):
    return s[:len(prefix)] == prefix
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.style as stylem
import matplotlib as mpl
plt.style.use('rose-pine-moon')
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
                if str(source) < str(destination):  # Process each edge only once
                    path = shortest_paths[source][destination]
                    traffic_flow = -1 / len(path)  # Negative traffic flow
                    L += traffic_flow

        # Update the maximum traffic load (Lmax) if necessary
        if L < Lmax:
            Lmax = L

        # Store the traffic load value
        traffic_loads.append(L)

    # Plot the traffic load without showing the percentage of node destruction
    plt.plot(range(0, 101, 10), traffic_loads, marker='o')
    plt.title("Traffic Load vs. Percentage of Node Destruction")
    plt.xlabel("Percentage of Node Destruction")
    plt.ylabel("Traffic Load (L)")
    plt.grid(True)
    plt.show()

    # Print the maximum traffic load observed
    print(f"\nMaximum Traffic Load (Lmax): {-Lmax}")
def calculate_traffic_load(G, plot_name):
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
                if str(source) < str(destination):  # Process each edge only once
                    path = shortest_paths[source][destination]
                    traffic_flow = -1 / len(path)  # Negative traffic flow
                    L += traffic_flow

        # Update the maximum traffic load (Lmax) if necessary
        if L < Lmax:
            Lmax = L

        # Store the traffic load value
        traffic_loads.append(L)

    # Plot the traffic load without showing the percentage of node destruction
    plt.plot(range(0, 101, 10), traffic_loads, marker='o')
    plt.title(f"Traffic Load vs. Percentage of Node Destruction ({plot_name})")
    plt.xlabel("Percentage of Node Destruction")
    plt.ylabel("Traffic Load (L)")
    plt.grid(True)
    plt.show()

    # Print the maximum traffic load observed
    print(f"\nMaximum Traffic Load ({plot_name}): {-Lmax}")





def generate_two_hub_base_map(base_network, hub_network, p_out):
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

    for j, i in zip(relabeled_hub_network.nodes(),base_network.nodes()):
        # With probability p_out, connect nodes from different networks
        if random.random() < p_out:
            G.add_edge(i, j)
    return G

    return G

import random

def rewire_generate_two_level_network(base_network, hub_network, p_out, p_rewire):
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
                # With probability p_rewire, rewire the edge to a random node in the opposite network
                if random.random() < p_rewire:
                    opposite_network_nodes = list(set(G.nodes()) - set(base_network.nodes()) - set(relabeled_hub_network.nodes()))
                    if opposite_network_nodes:
                        k = random.choice(opposite_network_nodes)
                        G.add_edge(i, k)
                else:
                    G.add_edge(i, j)

    return G
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
def calculate_traffic_load_map_level_hub_attack(base_network, hub_network, p_out):
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

    # Set the initial maximum traffic load to 0 for each network
    Lmax_base=0


    # List to store the corresponding traffic loads for each network
    traffic_loads_base = []
    traffic_loads_hub = []

    # Simulate node destruction and calculate traffic load for different percentages for each network
    num_iterations = 100
    for percent in range(0, 101, 10):  # Vary the percentage from 0 to 100 with a step of 10
        # Calculate the number of nodes to destroy based on the percentage for each network

        num_nodes_to_destroy_hub = int(percent / 100 * len(hub_network))

        # Create copies of the base and hub networks before node destruction
        base_network_copy = base_network.copy()
        hub_network_copy = hub_network.copy()



        # Perform node destruction on the hub network
        for i in range(num_nodes_to_destroy_hub):
            # Check if the hub network is empty
            if len(hub_network_copy) == 0:
                break

            # Select a random node to destroy
            node_to_destroy = random.choice(list(hub_network_copy.nodes()))

            # Remove the node and its associated edges
            hub_network_copy.remove_node(node_to_destroy)

        # Create a new graph by combining the remaining nodes from the base and hub networks
        G = nx.Graph()

        # Add nodes from the remaining base and hub networks
        G.add_nodes_from(base_network_copy.nodes())
        G.add_nodes_from(nx.relabel_nodes(hub_network_copy, {n: f'a{n}' for n in hub_network_copy.nodes()}).nodes())

        # Add edges within the remaining base and hub networks
        G.add_edges_from(base_network_copy.edges())
        G.add_edges_from(nx.relabel_nodes(hub_network_copy, {n: f'a{n}' for n in hub_network_copy.nodes()}).edges())

        # Add edges between the remaining base and hub networks
        for i in base_network_copy.nodes():
            for j in nx.relabel_nodes(hub_network_copy, {n: f'a{n}' for n in hub_network_copy.nodes()}).nodes():
                # With probability p_out, connect nodes from different networks
                if random.random() < p_out:
                    G.add_edge(i, j)
        network=generate_two_hub_base_map(base_network_copy,hub_network_copy,5)
        # Calculate shortest paths from source to destination for each network
        shortest_paths_base = nx.shortest_path(network)


        # Calculate the traffic load (L) by equally distributing traffic flow on shortest paths for the base network
        L_base = 0
        for source in shortest_paths_base:
            for destination in shortest_paths_base[source]:
                if str(source) < str(destination):  # Process each edge only once
                    path = shortest_paths_base[source][destination]
                    traffic_flow = -1 / len(path)  # Negative traffic flow
                    L_base += traffic_flow

        # Update the maximum traffic load (Lmax) for the base network if necessary
        if L_base < Lmax_base:
            Lmax_base = L_base

        # Store the traffic load value for the base network
        traffic_loads_base.append(L_base)



    # Plot the traffic load without showing the percentage of node destruction for each network
    plt.style.use('rose-pine-moon')
    plt.plot(range(0, 101, 10), traffic_loads_base, marker='o', label=' Network Hub Attack')
    plt.title("Traffic Load vs. Percentage of Node Destruction")
    plt.xlabel("Percentage of Node Destruction")
    plt.ylabel("Traffic Load (L)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Return the maximum traffic load observed for each network
    return -Lmax_base
def calculate_traffic_load_two_level_hub_attack(base_network, hub_network, p_out):
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

    # Set the initial maximum traffic load to 0 for each network
    Lmax_base=0


    # List to store the corresponding traffic loads for each network
    traffic_loads_base = []
    traffic_loads_hub = []

    # Simulate node destruction and calculate traffic load for different percentages for each network
    num_iterations = 100
    for percent in range(0, 101, 10):  # Vary the percentage from 0 to 100 with a step of 10
        # Calculate the number of nodes to destroy based on the percentage for each network

        num_nodes_to_destroy_hub = int(percent / 100 * len(hub_network))

        # Create copies of the base and hub networks before node destruction
        base_network_copy = base_network.copy()
        hub_network_copy = hub_network.copy()



        # Perform node destruction on the hub network
        for i in range(num_nodes_to_destroy_hub):
            # Check if the hub network is empty
            if len(hub_network_copy) == 0:
                break

            # Select a random node to destroy
            node_to_destroy = random.choice(list(hub_network_copy.nodes()))

            # Remove the node and its associated edges
            hub_network_copy.remove_node(node_to_destroy)

        # Create a new graph by combining the remaining nodes from the base and hub networks
        G = nx.Graph()

        # Add nodes from the remaining base and hub networks
        G.add_nodes_from(base_network_copy.nodes())
        G.add_nodes_from(nx.relabel_nodes(hub_network_copy, {n: f'a{n}' for n in hub_network_copy.nodes()}).nodes())

        # Add edges within the remaining base and hub networks
        G.add_edges_from(base_network_copy.edges())
        G.add_edges_from(nx.relabel_nodes(hub_network_copy, {n: f'a{n}' for n in hub_network_copy.nodes()}).edges())

        # Add edges between the remaining base and hub networks
        for i in base_network_copy.nodes():
            for j in nx.relabel_nodes(hub_network_copy, {n: f'a{n}' for n in hub_network_copy.nodes()}).nodes():
                # With probability p_out, connect nodes from different networks
                if random.random() < p_out:
                    G.add_edge(i, j)
        network=generate_two_level_network(base_network_copy,hub_network_copy,5)
        # Calculate shortest paths from source to destination for each network
        shortest_paths_base = nx.shortest_path(network)


        # Calculate the traffic load (L) by equally distributing traffic flow on shortest paths for the base network
        L_base = 0
        for source in shortest_paths_base:
            for destination in shortest_paths_base[source]:
                if str(source) < str(destination):  # Process each edge only once
                    path = shortest_paths_base[source][destination]
                    traffic_flow = -1 / len(path)  # Negative traffic flow
                    L_base += traffic_flow

        # Update the maximum traffic load (Lmax) for the base network if necessary
        if L_base < Lmax_base:
            Lmax_base = L_base

        # Store the traffic load value for the base network
        traffic_loads_base.append(L_base)



    # Plot the traffic load without showing the percentage of node destruction for each network
    plt.style.use('rose-pine-moon')
    plt.plot(range(0, 101, 10), traffic_loads_base, marker='o', label=' Network Hub Attack')
    plt.title("Traffic Load vs. Percentage of Node Destruction")
    plt.xlabel("Percentage of Node Destruction")
    plt.ylabel("Traffic Load (L)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Return the maximum traffic load observed for each network
    return -Lmax_base
def calculate_traffic_load_two_level(base_network, hub_network, p_out):
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

    # Set the initial maximum traffic load to 0 for each network
    Lmax_base = 0
    Lmax_hub = 0

    # List to store the corresponding traffic loads for each network
    traffic_loads_base = []
    traffic_loads_hub = []

    # Simulate node destruction and calculate traffic load for different percentages for each network
    num_iterations = 100
    for percent in range(0, 101, 10):  # Vary the percentage from 0 to 100 with a step of 10
        # Calculate the number of nodes to destroy based on the percentage for each network
        num_nodes_to_destroy_base = int(percent / 100 * len(base_network))
        num_nodes_to_destroy_hub = int(percent / 100 * len(hub_network))

        # Create copies of the base and hub networks before node destruction
        base_network_copy = base_network.copy()
        hub_network_copy = hub_network.copy()

        # Perform node destruction on the base network
        for i in range(num_nodes_to_destroy_base):
            # Check if the base network is empty
            if len(base_network_copy) == 0:
                break

            # Select a random node to destroy
            node_to_destroy = random.choice(list(base_network_copy.nodes()))

            # Remove the node and its associated edges
            base_network_copy.remove_node(node_to_destroy)

        # Perform node destruction on the hub network
        for i in range(num_nodes_to_destroy_hub):
            # Check if the hub network is empty
            if len(hub_network_copy) == 0:
                break

            # Select a random node to destroy
            node_to_destroy = random.choice(list(hub_network_copy.nodes()))

            # Remove the node and its associated edges
            hub_network_copy.remove_node(node_to_destroy)

        # Create a new graph by combining the remaining nodes from the base and hub networks
        G = nx.Graph()

        # Add nodes from the remaining base and hub networks
        G.add_nodes_from(base_network_copy.nodes())
        G.add_nodes_from(nx.relabel_nodes(hub_network_copy, {n: f'a{n}' for n in hub_network_copy.nodes()}).nodes())

        # Add edges within the remaining base and hub networks
        G.add_edges_from(base_network_copy.edges())
        G.add_edges_from(nx.relabel_nodes(hub_network_copy, {n: f'a{n}' for n in hub_network_copy.nodes()}).edges())

        # Add edges between the remaining base and hub networks
        for i in base_network_copy.nodes():
            for j in nx.relabel_nodes(hub_network_copy, {n: f'a{n}' for n in hub_network_copy.nodes()}).nodes():
                # With probability p_out, connect nodes from different networks
                if random.random() < p_out:
                    G.add_edge(i, j)

        # Calculate shortest paths from source to destination for each network
        shortest_paths_base = nx.shortest_path(base_network_copy)
        shortest_paths_hub = nx.shortest_path(hub_network_copy)

        # Calculate the traffic load (L) by equally distributing traffic flow on shortest paths for the base network
        L_base = 0
        for source in shortest_paths_base:
            for destination in shortest_paths_base[source]:
                if source < destination:  # Process each edge only once
                    path = shortest_paths_base[source][destination]
                    traffic_flow = -1 / len(path)  # Negative traffic flow
                    L_base += traffic_flow

        # Update the maximum traffic load (Lmax) for the base network if necessary
        if L_base < Lmax_base:
            Lmax_base = L_base

        # Store the traffic load value for the base network
        traffic_loads_base.append(L_base)

        #Calculate the traffic load (L) by equally distributing traffic flow on shortest paths for the hub network
        L_hub = 0
        for source in shortest_paths_hub:
            for destination in shortest_paths_hub[source]:
                if source < destination:  # Process each edge only once
                    path = shortest_paths_hub[source][destination]
                    traffic_flow = -1 / len(path)  # Negative traffic flow
                    L_hub += traffic_flow

        # Update the maximum traffic load (Lmax) for the hub network if necessary
        if L_hub < Lmax_hub:
            Lmax_hub = L_hub

        # Store the traffic load value for the hub network
        traffic_loads_hub.append(L_hub)

    # Plot the traffic load without showing the percentage of node destruction for each network
    plt.style.use('rose-pine-moon')
    plt.plot(range(0, 101, 10), traffic_loads_base, marker='o', label='Base Network')
    plt.plot(range(0, 101, 10), traffic_loads_hub, marker='o', label='Hub Network')
    plt.title("Traffic Load vs. Percentage of Node Destruction")
    plt.xlabel("Percentage of Node Destruction")
    plt.ylabel("Traffic Load (L)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Return the maximum traffic load observed for each network
    return -Lmax_base, -Lmax_hub
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
# Scenario 1
base_network  = nx.hypercube_graph(5)      # Example base network
hub_network = nx.hypercube_graph(2)          # Example hub network
p_out = 2                       # Probability of an edge between the networks

network = generate_two_level_network(base_network, hub_network, p_out)
calculate_traffic_load_two_level(base_network, hub_network, p_out)
calculate_traffic_load(base_network,"base scenario 1 " )
calculate_traffic_load( hub_network,"hub scenario 1" )
calculate_traffic_load( network,"Whole network scenario 1" )

# Visualize the network
#pos = nx.spring_layout(network)
#nx.draw(network, pos, with_labels=True)
#plt.show()

#calculate_traffic_load(network,"")

# Scenario 1 -reverse
hub_network  = nx.hypercube_graph(5)      # Example base network
base_network = nx.hypercube_graph(2)
p_out = 2                       # Probability of an edge between the networks

network = generate_two_level_network(base_network, hub_network, p_out)
calculate_traffic_load_two_level(base_network, hub_network, p_out)
calculate_traffic_load( network,"Reverse of network scenario 1" )
#-----------------------

# Scenario 2
base_network  = nx.hypercube_graph(2)      # Example base network
hub_network = tetrahedron_graph()          # Example hub network
p_out = 2                       # Probability of an edge between the networks

network = generate_two_level_network(base_network, hub_network, p_out)
calculate_traffic_load( network," network scenario 2" )
#calculate_traffic_load(network,"")

# Visualize the network
#pos = nx.spring_layout(network)
#nx.draw(network, pos, with_labels=True)
#plt.show()

# Scenario 2-reverse
base_network  = nx.hypercube_graph(2)      # Example base network
hub_network = tetrahedron_graph()           # Example hub network
p_out = 2                       # Probability of an edge between the networks

#network = generate_two_level_network(base_network, hub_network, p_out)
#calculate_traffic_load(network,"" )
network = generate_two_hub_base_map(base_network, hub_network, p_out)
calculate_traffic_load( network," reverse of scenario 2" )


# Visualize the network
pos = nx.spring_layout(network)
#nx.draw(network, pos, with_labels=True)
#plt.show()


# Scenario 3
base_network  = cube_graph(3)      # Example base network
hub_network = tetrahedron_graph()          # Example hub network
p_out = 2                       # Probability of an edge between the networks

network = generate_two_level_network(base_network, hub_network, p_out)
#calculate_traffic_load_two_level(base_network, hub_network, p_out)

# Visualize the network
#pos = nx.spring_layout(network)
#nx.draw(network, pos, with_labels=True)
#plt.show()


# Scenario 4
base_network  =nx.cycle_graph(30)
#  triangle graph
G10 = nx.Graph()
G10.add_edge(1, 2)
G10.add_edge(2, 3)
G10.add_edge(3, 1)

hub_network = G10       # Example hub network
p_out = 2                       # Probability of an edge between the networks

network = rewire_generate_two_level_network(base_network, hub_network, p_out,0.5)
network_anti_rewire = generate_two_level_network(base_network, hub_network, p_out)
calculate_traffic_load_two_level(base_network, hub_network, p_out)
calculate_traffic_load(network,"rewire")
calculate_traffic_load(network_anti_rewire,"network_anti_rewire")
# Visualize the network
pos = nx.spring_layout(network)
nx.draw(network, pos, with_labels=True)
plt.show()



# Scenario 4-2
base_network  =nx.cycle_graph(30)
#  triangle graph
G10 = nx.Graph()
G10.add_edge(1, 2)
G10.add_edge(2, 3)
G10.add_edge(3, 1)

hub_network = G10       # Example hub network
p_out = 0.33                      # Probability of an edge between the networks

network = generate_two_level_network(base_network, hub_network, p_out)
#calculate_traffic_load_two_level(base_network, hub_network, p_out)
calculate_traffic_load_two_level_hub_attack(base_network, hub_network, p_out)
# Visualize the network
pos = nx.spring_layout(network)
nx.draw(network, pos, with_labels=True)
plt.show()# Scenario 4-3
base_network  =nx.cycle_graph(30)
#  triangle graph
G10 = nx.Graph()
G10.add_edge(1, 2)
G10.add_edge(2, 3)
G10.add_edge(3, 1)

hub_network = G10       # Example hub network
p_out = 0.33                      # Probability of an edge between the networks

network = generate_two_hub_base_map(base_network, hub_network, 0.33)
calculate_traffic_load_map_level_hub_attack(base_network, hub_network, 0.33)
# Visualize the network
pos = nx.spring_layout(network)
nx.draw(network, pos, with_labels=True)

plt.show()

