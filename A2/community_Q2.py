import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import community as community_louvain
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering
from helper import *


def partitionVisualization(G, partition, methodName, colorSet, plotLabel=True):
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap(colorSet, max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=400,
                       cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5)
    if plotLabel:
        nx.draw_networkx_labels(G, pos)
    plt.title(methodName, fontsize=16)
    plt.show()


def clusteringByCentrality(G, method, nClusters=4, plotLabel=True):
    colorSets = ['Set2', 'Accent', 'Set3']
    if method=='louvain': # agglomerative clustering
        partition = community_louvain.best_partition(G)
        partitionVisualization(G, partition, 'Louvain clustering', colorSets[0], plotLabel)
        numClusters = max(list(partition.values()))+1
        return max(list(partition.values()))+1
    elif method=='fast_modularity': # agglomerative clustering
        partitionList = greedy_modularity_communities(G)
        partition = convertFrozenset2Dict(partitionList)
        partitionVisualization(G, partition, 'Fast modularity clustering', colorSets[1], plotLabel)
        numClusters = max(list(partition.values()))+1
        return max(list(partition.values()))+1
    elif method=='spectral': # spectral clustering
        A = nx.linalg.graphmatrix.adjacency_matrix(G)
        sc = SpectralClustering(n_clusters=nClusters, affinity='precomputed', n_init=100)
        sc.fit(A)
        partition = convertLabel2Dict(sc.labels_)
        partitionVisualization(G, partition, 'Spectral clustering', colorSets[2], plotLabel)
        

def communityDetection(G, plotLabel=True):
    numClusters1 = clusteringByCentrality(G, method='louvain', plotLabel=plotLabel)
    numClusters2 = clusteringByCentrality(G, method='fast_modularity', plotLabel=plotLabel)
    clusteringByCentrality(G, method='spectral', nClusters=max(numClusters1, numClusters2), plotLabel=plotLabel)
