import networkx as nx
import copy
from helper import *

def charikarDicts(G):
 
    S = copy.deepcopy(G)
    
    E = G.number_of_edges()
    N = G.number_of_nodes()
    
    nodes = {}
    best_avg = 0.0    
    iter = 0
    order = []
    
    for node, deg in G.degree():
        nodes[node] = S[node]

    nodesCopy = atlasViewCopy(nodes)

    while nodesCopy.keys():
        avg_degree = (2.0 * E)/N
        
        if best_avg <= avg_degree:
            best_avg = avg_degree
            best_iter = iter
            
        min_deg = N

        for n, neigh in nodesCopy.items():
            if min_deg >= len(neigh):
                min_deg = len(neigh)
                min_deg_node = n
        
        order.append(min_deg_node)

        for neigh in list(nodesCopy[min_deg_node].keys()):
            del nodesCopy[neigh][min_deg_node]
            
        del nodesCopy[min_deg_node]

        E -= min_deg
        N -= 1
        iter += 1
    
    S = copy.deepcopy(G)
    for i in range(best_iter):
        S.remove_node(order[i])
    return S, best_avg