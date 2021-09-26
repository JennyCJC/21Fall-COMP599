import numpy as np
import networkx as nx
import community as community_louvain
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import collections
from helper import *


def louvainClustering(G, colorSets=['Set2', 'Accent', 'Set3'], plotLabel=True):
    partition = community_louvain.best_partition(G)
    partitionVisualization(G, partition, 'Louvain clustering', colorSets[0], plotLabel)
    return max(list(partition.values()))+1, partition


def fastModularityClustering(G, colorSets=['Set2', 'Accent', 'Set3'], plotLabel=True):
    partitionList = greedy_modularity_communities(G)
    partition = convertFrozenset2Dict(partitionList)
    partitionVisualization(G, partition, 'Fast modularity clustering', colorSets[1], plotLabel)
    return max(list(partition.values()))+1, partition


def spectralClustering(G, nClusters=4, colorSets=['Set2', 'Accent', 'Set3'], plotLabel=True):
    # A = nx.linalg.graphmatrix.adjacency_matrix(G)
    csr = nx.convert_matrix.to_scipy_sparse_matrix(G)
    sc = SpectralClustering(n_clusters=nClusters, affinity='precomputed', n_init=1000)
    sc.fit(csr)
    partition = convertLabel2Dict(sc.labels_)
    partitionVisualization(G, partition, 'Spectral clustering', colorSets[2], plotLabel)
    return max(list(partition.values()))+1, partition
        

def communityDetection(G, plotLabel=True):

    numLouvainClusters, labelLouvain = louvainClustering(G, plotLabel=plotLabel)
    numFMClusters, labelFM = fastModularityClustering(G, plotLabel=plotLabel)
    numSpectralClusters, labelSpectral = spectralClustering(G, nClusters=int((numLouvainClusters + numFMClusters)/2), plotLabel=plotLabel)
    return ['Louvain', labelLouvain], ['Fast Modularity', labelFM], ['Spectral', labelSpectral]


def NMI(groundTruth, Prediction):
    return metrics.normalized_mutual_info_score(groundTruth, Prediction)


def ARI(groundTruth, Prediction):
    return metrics.adjusted_rand_score(groundTruth, Prediction)


def evaluatePerformance(groundTruth, prediction):
    print(prediction[0] + ' ARI score: ')
    print(metrics.adjusted_rand_score(groundTruth, prediction[1]))

    print(prediction[0] + ' NMI score: ')
    print(metrics.normalized_mutual_info_score(groundTruth, prediction[1]))

def overallPerformance(graph, Predictions):
    orderedPredictions = []
    for pred in Predictions:
        orderedPredictions.append([pred[0], getLabels(pred[1])])
    
    truthDict = dict(graph.nodes(data='value', default=1))
    truthLabels = getLabels(truthDict)

    for pair in orderedPredictions:
        evaluatePerformance(truthLabels, pair)