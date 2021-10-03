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
    return list(collections.OrderedDict(sorted(convertDictKeyToInt(dictList).items())).values())