import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os,glob
from preprocess import * 

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

def networkVisualization(G):
    nx.draw_kamada_kawai(G)
    plt.show()
    

def showTopSubgraphs(topSubgraphs):
    nx.draw(topSubgraphs[0][1], node_size=500, node_color='lightcoral')
    plt.title('Densest subgraph')
    print(topSubgraphs[0][1].degree)
    plt.show()
    
    nx.draw(topSubgraphs[1][1], node_size=500, node_color='cornflowerblue')
    plt.title('Densest subgraph')
    print(topSubgraphs[1][1].degree)
    plt.show()
    
    nx.draw(topSubgraphs[2][1], node_size=500, node_color='slategray')
    plt.title('Densest subgraph')
    print(topSubgraphs[2][1].degree)
    plt.show()
    
    print(topSubgraphs[0][1].number_of_nodes())
    print(topSubgraphs[0][1].nodes())
    print(topSubgraphs[1][1].number_of_nodes())
    print(topSubgraphs[1][1].nodes())
    print(topSubgraphs[2][1].number_of_nodes())
    print(topSubgraphs[2][1].nodes())
    #networkVisualization(diffG)


def generateEdgelists (folder_path):

    if not os.path.isdir(str(folder_path + '/asdsEdgelists')):
        os.makedirs(folder_path+'/asdsEdgelists')

    if not os.path.isdir(str(folder_path + '/tdsEdgelists')):
        os.makedirs(folder_path+'/tdsEdgelists')

    asds = load_graphs(folder_path, "asd")
    tds = load_graphs(folder_path, "td")
    
    asds = np.swapaxes(asds, 0, 2)
    tds = np.swapaxes(tds, 0, 2)

    i = 0;

    for A in asds:
        g = nx.convert_matrix.from_numpy_matrix(A)
        nx.readwrite.edgelist.write_edgelist(g, str(folder_path+'/asdsEdgelists/'+str(i)+'.txt'), data=False)
        i+=1
    
    i = 0;
    for A in tds:
        g = nx.convert_matrix.from_numpy_matrix(A)
        nx.readwrite.edgelist.write_edgelist(g, str(folder_path+'/tdsEdgelists/'+str(i)+'.txt'), data=False)
        i+=1
    