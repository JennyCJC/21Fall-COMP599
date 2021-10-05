import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import collections

#Q1 Helper functions
def convertIndexToNames(index):
    dataFrame = pd.read_table("email-Enron/addresses-email-Enron.txt", 
                names=['Index', 'Email Address'])
    mostImportantEmail = dataFrame[dataFrame['Index'].isin(index)]
    mostImportantEmail = mostImportantEmail['Email Address']
    return np.asarray(mostImportantEmail)


def postProcessing(dictCentrality):
    rankings = np.asarray(list(dictCentrality.values()))
    mostPopularIdx = np.argsort((-1) * rankings) # rank from highest to lowest
    mostPopularIdx = mostPopularIdx[0:5]
    mostImportantEmail = convertIndexToNames(mostPopularIdx)
    highestCentrality = rankings[mostPopularIdx]
    return (mostImportantEmail, highestCentrality)

def printRankingResults(centralityMeasure, mostImportantEmail, highestCentrality):
    print(centralityMeasure + ': ')
    for i in range(5):
        print('Ranking: ' + str(i+1) + ';   Email: ' + mostImportantEmail[i] + 
        ';  ' + centralityMeasure + ':  '+ str(highestCentrality[i]))
    print('')


#Q2 Helper functions
def convertFrozenset2Dict(partitionList):
    sets = [list(x) for x in partitionList]
    partition = {}
    for clusterIdx in range(len(sets)):
        for node in sets[clusterIdx]:
            partition[node] = clusterIdx
    return partition


def convertLabel2Dict(G, labels):    
    partition = {}
    nodeNames = list(G.nodes)
    for nodeIdx in range(len(labels)):
        partition[nodeNames[nodeIdx]] = labels[nodeIdx]
    return partition


def convertDist2List(partition):
    keys, values = partition.items()
    keys = list(keys)
    numClusters = max(list(values)) + 1
    communities = []
    for cluster in range(numClusters):
        print(list(values)==cluster)
        communities.append(keys[values==cluster])
    return communities


def extractGCC(G):
    CC = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(CC[0])
    return G0

def partitionVisualization(G, partition, methodName, colorSet, plotLabel=True, largeNetwork=False):
    if not largeNetwork:
        pos = nx.spring_layout(G)
        # color the nodes according to their partition
        cmap = cm.get_cmap(colorSet, max(partition.values()) + 1)
        nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=200, alpha=0.5,
                        cmap=cmap, node_color=list(partition.values()))
        nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5)
        if plotLabel:
            nx.draw_networkx_labels(G, pos)
        plt.title(methodName, fontsize=16)
        plt.show()
    else:
        largeNetworkPartitionVisual(G, partition, methodName)


def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1


def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0


def get_color(i, r_off=1, g_off=1, b_off=1):
    '''Assign a color to a vertex.'''
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)      


def largeNetworkPartitionVisual(G, communities, methodName):
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update({'figure.figsize': (15, 10)})
        plt.style.use('dark_background')

        # Set node and edge communities
        set_node_community(G, communities)
        set_edge_community(G)

        # Set community color for internal edges
        external = [(v, w) for v, w in list(G.edges) if G.edges[v, w]['community'] == 0]
        internal = [(v, w) for v, w in list(G.edges) if G.edges[v, w]['community'] > 0]
        internal_color = ["black" for e in internal]
        node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]

        # external edges
        pos = nx.spring_layout(G, k=0.2)
        nx.draw_networkx(G, pos=pos, node_size=0, edgelist=external, edge_color="silver",
        node_color=node_color, alpha=0.2, with_labels=False)

        # internal edges
        nx.draw_networkx(G, pos=pos, edgelist=internal, edge_color=internal_color,
        node_color=node_color, alpha=0.4, with_labels=False)
        plt.title(methodName, fontsize=16)
        plt.show()
        

def convertDictKeyToInt(dictList):
    intDictList = {}
    for key in dictList:
        intDictList[int(key)] = dictList[key]
    return intDictList


def getLabels(dictList):
    return list(collections.OrderedDict(sorted(dictList.items())).values())

def getSetsFromLabels(labels):
    communitySets = []
    communityNum = max(list(labels.values()))

    for i in range(communityNum+1):
        communitySets.append(set())

    for key, value in labels.items():
        communitySets[int(value)].add(key)

    return communitySets

def getLabelDictFromCommunities(communities):
    labelDict = {}
    ct = 0
    for community in communities:
        for node in community:
            labelDict[node] = ct
        ct +=1
    return labelDict

def getUniqueCommunities(graph):
    communities = []
    nodeDict = dict(graph.nodes(data='community', default={}))

    for value in nodeDict.values():
        if value not in communities:
            communities.append(value)
    return communities