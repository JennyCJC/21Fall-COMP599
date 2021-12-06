import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

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


def edgelist_format(path, outfileName):
    graph = loadGraph(path)
    features = {}
    for i in range(graph.number_of_nodes()):
        features[str(i)] = str(graph.degree[i])
    
    edgelist = list(nx.convert.to_edgelist(graph))
    edges = []
    for i in range(len(edgelist)):
        edges.append([edgelist[i][0], edgelist[i][1]])
    
    graph_dict = {}
    graph_dict['edges'] = edges
    graph_dict['features'] = features
    
    print(outfileName)
    with open(outfileName, 'w') as outfile:
        json.dump(graph_dict, outfile, ensure_ascii=False)   
    
    
def edgelist_files(path):
    outIdx = 0
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path,file)):
            outfileName = os.path.join(path, "edgelist/"+ str(outIdx) + ".json")
            edgelist_format(os.path.join(path,file), outfileName)
            outIdx = outIdx + 1


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
    
    
def load_graph2vec_features(category):
    path = "/Users/ann/Desktop/Fall2021/COMP599/21Fall-COMP599/project/graph2vec/features"
    features_asd = pd.read_csv(os.path.join(path, category+'_asd.csv'), header=None)
    features_td = pd.read_csv(os.path.join(path, category+'_td.csv'), header=None)
    features = np.concatenate((features_asd.to_numpy()[1:, 1:], features_td.to_numpy()[1:, 1:]), axis=0)
    labels = np.concatenate((np.ones(features_asd.shape[0]-1), np.zeros(features_td.shape[0]-1)), axis=0)
    return features, labels

    