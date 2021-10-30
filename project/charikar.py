import networkx as nx
import copy

def charikarDicts(G):
 
    S = copy.deepcopy(G)
    
    #E = sum((G.degree(weight = attr)).values())
    
    E = G.number_of_edges()
    N = G.number_of_nodes()
    
    nodes = {}
    best_avg = 0.0    
    iter = 0
    order = []
    
    for node, deg in G.degree():
        nodes[node] = S[node]
        
    while nodes.keys():
        avg_degree = (2.0 * E)/N
        
        if best_avg <= avg_degree:
            best_avg = avg_degree
            best_iter = iter
            
        min_deg = N
        for n, neigh in nodes.items():
            if min_deg > len(neigh):
                min_deg = len(neigh)
                min_deg_node = n
                
        #print min_deg, min_deg_node
        
        order.append(min_deg_node)
            
        for neigh in nodes[min_deg_node].keys():
            #E -= nodes[neigh][min_deg_node] if attr else 1
            #E -= 1
            del nodes[neigh][min_deg_node] 
            
        del nodes[min_deg_node]
        E -= min_deg
        N -= 1
        iter += 1
    
    S = copy.deepcopy(G)
    for i in range(best_iter):
        S.remove_node(order[i])
    return S, best_avg
    

def charikarLinear(G):

    E = G.number_of_edges()
    N = G.number_of_nodes()
    best_avg = 0.0
    best_iter = 0

    # trivial case, empty graph
    if E == 0:
        return G

    nodes_by_degree = {i: dict() for i in range(len(G))}
    degree_by_node = {}
    order = []
    neighbors = {}

    for node in G:
        deg = G.degree(node)
        nodes_by_degree[deg][node] = 1
        degree_by_node[node] = deg
        neighbors[node] = {neighbor: 1 for neighbor in G.neighbors_iter(node)}

    min_deg = 0
    while not nodes_by_degree[min_deg]:
        min_deg += 1

    for it in range(N - 1):
        # update best subgraph
        avg_degree = (2.0 * E) / N
        if best_avg <= avg_degree:
            best_avg = avg_degree
            best_iter = it
        # pick a node with minimum degree for deletion
        min_deg_node = nodes_by_degree[min_deg].iterkeys().next()
        order.append(min_deg_node)
        del nodes_by_degree[min_deg][min_deg_node]
        E -= min_deg
        N -= 1
        # update neighbors
        # decrease the degree of all neighbors of min_deg_node
        # by one
        for neighbor in neighbors[min_deg_node]:
            del nodes_by_degree[degree_by_node[neighbor]][neighbor]
            degree_by_node[neighbor] -= 1
            nodes_by_degree[degree_by_node[neighbor]][neighbor] = 1
            del neighbors[neighbor][min_deg_node]

        if min_deg > 0 and nodes_by_degree[min_deg - 1]:
            min_deg -= 1
        else:
            while not nodes_by_degree[min_deg]:
                min_deg += 1

    S = nx.Graph()
    to_ignore = set(order[:best_iter])
    for u, v in G.edges_iter():
        if u not in to_ignore and v not in to_ignore:
            S.add_edge(u, v)
    return S, best_avg
