import numpy as np
import networkx as nx
import community as community_louvain
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from helper import *
import copy


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
    partition = convertLabel2Dict(G, sc.labels_)
    partitionVisualization(G, partition, 'Spectral clustering', colorSets[2], plotLabel)
    return max(list(partition.values()))+1, partition
        

def communityDetection(G, plotLabel=True):

    numLouvainClusters, labelLouvain = louvainClustering(G, plotLabel=plotLabel)
    numFMClusters, labelFM = fastModularityClustering(G, plotLabel=plotLabel)
    # numSpectralClusters, labelSpectral = spectralClustering(G, nClusters=int((numLouvainClusters + numFMClusters)/2), plotLabel=plotLabel)
    # , ['Spectral', labelSpectral]
    
    return ['Louvain', labelLouvain], ['Fast Modularity', labelFM]


def evaluatePerformance(groundTruth, prediction, graph):
    print(prediction[0] + ' ARI score: ')
    print(metrics.adjusted_rand_score(groundTruth, prediction[1]))

    print(prediction[0] + ' NMI score: ')
    print(metrics.normalized_mutual_info_score(groundTruth, prediction[1]))

    print(prediction[0] + ' Q-Modularity score: ')
    print(nx.algorithms.community.modularity(graph, getSetsFromLabels(prediction[2])))


def overallPerformance(graph, Predictions, removeUnlabeled=[], truthLabels=[]):
    if len(removeUnlabeled) > 0:
        orginalPrediction = copy.deepcopy(Predictions)
        for nonLabeled in removeUnlabeled:
            for pred in Predictions:
                pred[1].pop(int(nonLabeled))

    orderedPredictions = []
    for i in range(len(Predictions)):
        orderedPredictions.append([Predictions[i][0], getLabels(Predictions[i][1]), Predictions[i][1] if len(removeUnlabeled) == 0 else orginalPrediction[i][1]])

    if len(truthLabels) <= 0:
        truthDict = dict(graph.nodes(data='value', default=1))
        truthLabels = getLabels(truthDict)

    for pair in orderedPredictions:
        evaluatePerformance(truthLabels, pair, graph)