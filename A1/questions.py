import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy import sparse
from helpers import *

def degreeDistribution(simpleGraph):
    #calculate degree and frequency
    sumG = simpleGraph.sum(axis=1)
    sumArray = np.squeeze(np.asarray(sumG))
    degreeData = np.array(freq(sumArray))

    #show data in plot
    plt.loglog(degreeData[:, 0],degreeData[:, 1])
    plt.show()

# ten = [10]
# powerDegree = [0.5, 1.0, 1.5, 2.0]
# powerFreq = [1, 2, 3, 4]

# np.polyfit(np.power(ten, powerDegree), np.power(ten, powerFreq), )
# print(np.power(ten, powerDegree))
# print(stats.relfreq(sumArray, numbins=100))


def clusterCoefDistribution(simpleGraph):
    # calculate degree 
    sumG = simpleGraph.sum(axis=1)
    degree = np.squeeze(np.asarray(sumG))

    # calculate local clustering coefficient & frequency
    A3 = simpleGraph * simpleGraph * simpleGraph
    numClosedPath = np.diagonal(A3.toarray())
    numPath = degree * (degree-1)
    finiteIdx = np.nonzero(numPath)
    finiteC = numClosedPath[finiteIdx] / numPath[finiteIdx]
    uniqueC, counts = np.unique(finiteC, return_counts=True)
    freq = np.cumsum(counts) / sum(counts)

    # compute and print the average C
    avgC = np.mean(finiteC)
    print("Average clustering coefficient: " + str(avgC))

    # show the distribution in a plot
    plt.plot(uniqueC, freq)
    plt.xlim([0,2])
    plt.show()


def shortestPathDistribution(simpleGraph):
    # calculate shortest distances & the corresponding frequencies
    distance = sparse.csgraph.shortest_path(simpleGraph)
    uniqueDistance, counts = np.unique(distance, return_counts=True)
    finiteIdx = np.isfinite(uniqueDistance)
    uniqueDistance = uniqueDistance[finiteIdx]
    counts = counts[finiteIdx]
    freq = counts / sum(counts)

    # show the distribution in a plot
    plt.plot(uniqueDistance, freq)
    plt.show()


def connectivity(simpleGraph):
    # calculate and print the number of connected components
    numConnected, labels = sparse.csgraph.connected_components(simpleGraph, 
                directed=False, return_labels=True)
    print("Number of connected components: " + str(numConnected))

    # print the portion of nodes in the GCC
    numNodes, counts = np.unique(labels, return_counts=True)
    print("Number of nodes in the GCC: " + str(max(counts)))


def eigenvalueDistribution(simpleGraph):
    # calculate the eigenvalues of this graph
    laplacian = sparse.csgraph.laplacian(simpleGraph)
    laplacian = laplacian.asfptype().toarray()
    eval_max = sparse.linalg.eigs(laplacian, k=100, which='LM')
    eval_min = sparse.linalg.eigs(laplacian, k=100, which='SM')


def degreeCorrelation(simpleGraph):
    # calculate degree 
    sumG = simpleGraph.sum(axis=1)
    degree = np.squeeze(np.asarray(sumG))
    
    # construct the source degree vector & destination degree vector
    sourceNode = simpleGraph.nonzero()[0]
    destinationNode = simpleGraph.nonzero()[1]
    sourceDegree = [degree[node] for node in sourceNode]
    destinationDegree = [degree[node] for node in destinationNode]

    # report the overall correlation
    overallCorr = np.corrcoef(sourceDegree, destinationDegree)[0,1]
    print("Overall degree correlation: " + str(overallCorr))

    # plot degree of source VS degree of destination
    plt.scatter(sourceDegree, destinationDegree)
    plt.xlim(0, np.max(degree))
    plt.ylim(0, np.max(degree))
    plt.show()


def degreeClusterCoefRelation(simpleGraph):
    # calculate degree 
    sumG = simpleGraph.sum(axis=1)
    degree = np.squeeze(np.asarray(sumG))

    # calculate clustering coefficient
    A3 = simpleGraph * simpleGraph * simpleGraph
    numClosedPath = np.diagonal(A3.toarray())
    numPath = degree * (degree-1)
    finiteIdx = np.nonzero(numPath)
    finiteC = numClosedPath[finiteIdx] / numPath[finiteIdx]

    # plot degree VS clustering coefficient
    degree = degree[finiteIdx]
    plt.scatter(degree, finiteC)
    plt.show()








    


