import numpy as np
import networkx as nx
import community as community_louvain
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from helper import *


def louvainClustering(G, colorSets=['Set2', 'Accent', 'Set3'], plotLabel=True):
    partition = community_louvain.best_partition(G)
    print(sorted(int(partition)))
    partitionVisualization(G, partition, 'Louvain clustering', colorSets[0], plotLabel)
    return max(list(partition.values()))+1


def fastModularityClustering(G, colorSets=['Set2', 'Accent', 'Set3'], plotLabel=True):
    partitionList = greedy_modularity_communities(G)
    partition = convertFrozenset2Dict(partitionList)
    print(partition)
    partitionVisualization(G, partition, 'Fast modularity clustering', colorSets[1], plotLabel)
    return max(list(partition.values()))+1


def spectralClustering(G, nClusters=4, colorSets=['Set2', 'Accent', 'Set3'], plotLabel=True):
    # A = nx.linalg.graphmatrix.adjacency_matrix(G)
    csr = nx.convert_matrix.to_scipy_sparse_matrix(G)
    sc = SpectralClustering(n_clusters=nClusters, affinity='precomputed', n_init=1000)
    sc.fit(csr)
    partition = convertLabel2Dict(sc.labels_)
    partitionVisualization(G, partition, 'Spectral clustering', colorSets[2], plotLabel)
    return max(list(partition.values()))+1
        

def communityDetection(G, plotLabel=True):

    numLouvainClusters = louvainClustering(G, plotLabel=plotLabel)
    numFMClusters = fastModularityClustering(G, plotLabel=plotLabel)
    numSpectralClusters = spectralClustering(G, nClusters=int((numLouvainClusters + numFMClusters)/2), plotLabel=plotLabel)


def NMI(groundTruth, Prediction):
    return metrics.normalized_mutual_info_score(groundTruth, Prediction)

def ARI(groundTruth, Prediction):
    return metrics.adjusted_rand_score(groundTruth, Prediction)