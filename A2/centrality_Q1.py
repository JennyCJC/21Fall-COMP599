import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from helper import *


def degreeCentrality(G):
    G = nx.convert_matrix.from_scipy_sparse_matrix(G)
    dictDegreeCentrality = nx.algorithms.degree_centrality(G)
    mostImportantEmail, highestDegreeCentrality = postProcessing(dictDegreeCentrality)
    printRankingResults('Degree centrality', mostImportantEmail, highestDegreeCentrality)


def eigenvectorCentrality(G):
    G = nx.convert_matrix.from_scipy_sparse_matrix(G)
    dictEVCentrality = nx.algorithms.centrality.eigenvector_centrality(G)
    mostImportantEmail, highestEVCentrality = postProcessing(dictEVCentrality)
    printRankingResults('Eigenvector centrality', mostImportantEmail, highestEVCentrality)


def katzCentrality(G):
    # set alpha to be less than 1/eval_max
    laplacian = sparse.csgraph.laplacian(G).asfptype()
    laplacian = laplacian.todense()
    eval_max = sparse.linalg.eigs(laplacian, k=1, which='LM')
    eval_max = eval_max[0][0]
    alphaValue = 0.9 / eval_max.real

    G = nx.convert_matrix.from_scipy_sparse_matrix(G)
    dictKatsCentrality = nx.algorithms.centrality.katz_centrality(G, alpha=alphaValue)
    mostImportantEmail, highestKatsCentrality = postProcessing(dictKatsCentrality)
    printRankingResults('Kats centrality', mostImportantEmail, highestKatsCentrality)


def betweennessCentrality(G):
    G = nx.convert_matrix.from_scipy_sparse_matrix(G)
    dictBetweennessCentrality = nx.algorithms.centrality.betweenness_centrality(G)
    mostImportantEmail, highestBetweennessCentrality = postProcessing(dictBetweennessCentrality)
    printRankingResults('Betweenness centrality', mostImportantEmail, highestBetweennessCentrality)


def mostImportantNodes(G):
    degreeCentrality(G)
    eigenvectorCentrality(G)
    katzCentrality(G)
    #betweennessCentrality(G)

