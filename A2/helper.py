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

def partitionVisualization(G, partition, methodName, colorSet, plotLabel=True):
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap(colorSet, max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=200,
                       cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5)
    if plotLabel:
        nx.draw_networkx_labels(G, pos)
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


def removeUnlabeledCiteseerNodes(graph, testIndex, labelIndex):
    nodeList = list(range(len(labelIndex), graph.number_of_nodes()))
    
    for i in testIndex.astype(int):
        if i in nodeList:
            nodeList.remove(i)
    
    for i in nodeList:
        graph.remove_node(i)
    
    return graph