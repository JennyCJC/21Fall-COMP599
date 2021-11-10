import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def printTopINodesSubgraph(output, i, subgr, induced):
    if induced:
        # inducedSubgr = inducedSubgraph(subgr)
        # inducedSubgr.printTopI(output, i, false)
        print('not implemented yet')
    else:
        return (output + '_Top' + str(i), subgr)

def loadGraph(path):
    adjacency_matrix = np.loadtxt(path, dtype=float)
    graph = nx.convert_matrix.from_numpy_matrix(adjacency_matrix)
    return graph

def atlasViewCopy(atlasView):
    copy = {}

    for element in atlasView:
        elementValues = {}
        for el in atlasView[element]:
            elementValues[el] = atlasView[element][el]
        copy[element] = elementValues  
    
    return copy


def removeWeakConnections(g, currentSubgr, alpha):

    allowed = len(g.nodes())*alpha
    m = 0
    neighborsDict = {}
    minDeg = 0
    maxDeg = 0
    
    for node in currentSubgr:
        neighbors = g[node]

        neighborsCopy = atlasViewCopy(neighbors)
        
        m += len(neighborsCopy)
        if node in neighborsCopy:
            m += 1

        for el in currentSubgr:
            if el in neighborsCopy:
                neighborsCopy.pop(el)

        deg = len(neighborsCopy)
        m -= deg
        if deg not in neighborsDict:
            neighborsDict[deg] = set()
        neighborsDict[deg].add(node)

        if maxDeg < deg:
            maxDeg = deg
        if minDeg > deg:
            minDeg = deg
        

    m = m/2
    i = 0
    j = maxDeg

    # print(j)
    # print(neighborsDict)
    while i < allowed:
        i += 1
        x = next(iter(neighborsDict[j]))
        neighborsDict[j].remove(x)
        currentSubgr.remove_node(x)

        while (j not in neighborsDict or len(neighborsDict[j]) == 0) and j > 0:
            j -= 1
    
    g.remove_nodes_from(currentSubgr)

    return g, m
