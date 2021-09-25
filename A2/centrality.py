import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse.linalg
from scipy import sparse


def convertIndex2Names(index):
    dataFrame = pd.read_table("email-Enron/addresses-email-Enron.txt", 
                names=['Index', 'Email Address'])
    mostImportantEmail = dataFrame[dataFrame['Index'].isin(index)]
    mostImportantEmail = mostImportantEmail['Email Address']
    return np.asarray(mostImportantEmail)


def postProcessing(dictCentrality):
    rankings = np.asarray(list(dictCentrality.values()))
    mostPopularIdx = np.argsort((-1) * rankings) # rank from highest to lowest
    mostPopularIdx = mostPopularIdx[0:5]
    mostImportantEmail = convertIndex2Names(mostPopularIdx)
    highestCentrality = rankings[mostPopularIdx]
    return (mostImportantEmail, highestCentrality)


def printRankingResults(centralityMeasure, mostImportantEmail, highestCentrality):
    print(centralityMeasure + ': ')
    for i in range(5):
        print('Ranking: ' + str(i+1) + ';   Email: ' + mostImportantEmail[i] + 
        ';  ' + centralityMeasure + ':  '+ str(highestCentrality[i]))
    print('')


def degreeCentrality(graphData):
    graphData = nx.convert_matrix.from_scipy_sparse_matrix(graphData)
    dictDegreeCentrality = nx.algorithms.degree_centrality(graphData)
    mostImportantEmail, highestDegreeCentrality = postProcessing(dictDegreeCentrality)
    printRankingResults('Degree centrality', mostImportantEmail, highestDegreeCentrality)


def eigenvectorCentrality(graphData):
    graphData = nx.convert_matrix.from_scipy_sparse_matrix(graphData)
    dictEVCentrality = nx.algorithms.centrality.eigenvector_centrality(graphData)
    mostImportantEmail, highestEVCentrality = postProcessing(dictEVCentrality)
    printRankingResults('Eigenvector centrality', mostImportantEmail, highestEVCentrality)


def KatzCentrality(graphData):
    # set alpha to be less than 1/eval_max
    laplacian = sparse.csgraph.laplacian(graphData).asfptype()
    laplacian = laplacian.todense()
    eval_max = sparse.linalg.eigs(laplacian, k=1, which='LM')
    eval_max = eval_max[0][0]
    alphaValue = 0.9 / eval_max.real

    graphData = nx.convert_matrix.from_scipy_sparse_matrix(graphData)
    dictKatsCentrality = nx.algorithms.centrality.katz_centrality(graphData, alpha=alphaValue)
    mostImportantEmail, highestKatsCentrality = postProcessing(dictKatsCentrality)
    printRankingResults('Kats centrality', mostImportantEmail, highestKatsCentrality)


def betweennessCentrality(graphData):
    graphData = nx.convert_matrix.from_scipy_sparse_matrix(graphData)
    dictBetweennessCentrality = nx.algorithms.centrality.betweenness_centrality(graphData)
    mostImportantEmail, highestBetweennessCentrality = postProcessing(dictBetweennessCentrality)
    printRankingResults('Betweenness centrality', mostImportantEmail, highestBetweennessCentrality)


def mostImportantNodes(graphData):
    degreeCentrality(graphData)
    eigenvectorCentrality(graphData)
    KatzCentrality(graphData)
    #betweennessCentrality(graphData)



    