import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def printTopINodesSubgraph(output, i, subgr, induced):
    if induced:
        # inducedSubgr = inducedSubgraph(subgr)
        # inducedSubgr.printTopI(output, i, false)
    else:
        return (output + '_Top' + str(i), subgr)

def loadGraph(path):
    adjacency_matrix = np.loadtxt(path, dtype=int)
    graph = nx.convert_matrix.from_numpy_matrix(adjacency_matrix)
    return graph
