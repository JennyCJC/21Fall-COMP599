import networkx as nx
from networkx.classes.function import number_of_edges
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy


# Q1 helper functions

def dropLabels(G, dropPercentage, labelName):
    #maskedG = G.copy()
    maskedG = copy.deepcopy(G)
    numTotalNodes = G.number_of_nodes()
    numDropLabels = math.floor(numTotalNodes * dropPercentage)
    nodeOrder = list(maskedG.nodes)
    random.shuffle(nodeOrder)
    dropLabels = nodeOrder[0:numDropLabels]
    for node in dropLabels:
        del maskedG.nodes[node][labelName]
    return (maskedG, dropLabels)
    

def evaluateAccuracy(G, dropLabels, predictedLabel, labelName):
    nodeNames = list(G.nodes)
    predictIdx = np.isin(nodeNames, dropLabels)
    predictedLabel = np.array(predictedLabel)[predictIdx]
    trueLabel = nx.get_node_attributes(G, labelName)
    trueLabel = np.array(list(trueLabel.values()))[predictIdx]
    accuracy = sum(np.array(trueLabel)==np.array(predictedLabel)) / len(trueLabel)
    return round(accuracy, 4)


def plotAccuracy(dropLabelPercentage, accuracies, datasetName, method):
    plt.rcParams.update({'font.size': 14})
    p = plt.plot(np.array(dropLabelPercentage, dtype='|S6'), accuracies)
    plt.setp(p, color='g', linewidth=3.0)
    plt.xlabel('Percentage of dropped label')
    plt.ylabel('Accuracy')
    plt.title(datasetName + '(' + method + ')')
    plt.show()


def printNodeClassifResult(datasetName, method, droppedPercentage, accuracies):
    print(datasetName + ' using ' + method + ':')
    print('Dropped label: ', droppedPercentage)
    print('Accuracies: ', accuracies)


def getTrainIdx(G, testIdx):
    return [idx for idx in range(G.number_of_nodes()) if ~np.isin(idx, testIdx)]


def addLabel2Nodes(G, testIdx, orderedLabel):
    trainIdx = getTrainIdx(G, testIdx)
    for idx in range(len(trainIdx)):
        G.nodes[trainIdx[idx]]['label'] = orderedLabel[idx] 



# Q2 helper functions

def dropLinks(G, dropProportion=0.2):
    numDropEdges = math.floor(G.number_of_edges() * dropProportion)
    maskedG = copy.deepcopy(G)
    allEdges = list(G.edges)
    random.shuffle(allEdges)
    removeEdges = allEdges[0:numDropEdges]
    maskedG.remove_edges_from(removeEdges)
    return (maskedG, removeEdges)

def generateLinkFromCoefficient(preds, droppedLinks, threshold):
    predDict = {}
    for u, v, p in preds:
        predDict[(u, v)] = p

    predLinks = []
    for key, value in sorted(predDict.items(), key=lambda x: x[1]):
        # threshold is 0.025 for now
        if value > threshold:
            predLinks.append(key)

    truePositive = set(predLinks).intersection(droppedLinks)
    tpr = len(truePositive)/len(droppedLinks)
    fpr = (len(predLinks)-len(truePositive))/len(predDict)
    return fpr, tpr
    

def multiGraphToSimpleGraph(M):
    G = nx.Graph()
    for u,v,data in M.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if G.has_edge(u,v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G

def randomSelection(percentage, edges):
    random.shuffle(edges)
    selectedEdges = edges[0: int(len(edges)*percentage)]
    return selectedEdges

def findNonEdges(graph, percentage):
    number_of_edges = int((nx.classes.function.number_of_edges(graph))*percentage)
    nonEdgeList = []
    allNodes = graph.nodes
    while len(nonEdgeList) <= number_of_edges:
        n = random.sample(allNodes, k=2)
        if not graph.has_edge(n[0], n[1]):
            nonEdgeList.append((n[0], n[1]))
    
    return nonEdgeList