import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy import sparse
from helpers import *
import math

def networkPatterns(simpleGraph):
    degreeDistribution(simpleGraph)         #1A
    clusterCoefDistribution(simpleGraph)    #1B
    shortestPathDistribution(simpleGraph)   #1C
    connectivity(simpleGraph)               #1D
    eigenvalueDistribution(simpleGraph)     #1E
    degreeCorrelation(simpleGraph)          #1F
    degreeClusterCoefRelation(simpleGraph)  #1G


def degreeDistribution(simpleGraph):
    #calculate degree and frequency
    sumG = simpleGraph.sum(axis=1)
    sumArray = np.squeeze(np.asarray(sumG))
    degreeData = np.array(freq(sumArray))

    #log binning with bin size 15, but bin could also be removed if no data belongs to the bin
    degreeData = np.array(logBinning(15, degreeData))
    
    #find polynomial fit
    logX = list(map(math.log1p, degreeData[:, 0]))
    logY = list(map(math.log1p, degreeData[:, 1]))

    coefficients = np.polyfit(logX, logY, 1)
    poly = np.poly1d(coefficients)

    #show data in plot
    xValues = list(map(math.log1p, (np.arange(max(degreeData[:, 0])))))
    yValues = np.polyval(poly, xValues)
    plt.plot(list(map(math.exp, xValues)), list(map(math.exp, yValues)))
    
    plt.loglog(degreeData[:, 0],degreeData[:, 1], "o")
    plt.show()

    #return slope
    print('Slope for the degree distribution is: ' + str(coefficients[0]))

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


def create_BA_Graph(simpleGraph):
    numFinalNodes = np.shape(simpleGraph)[0]
    numAvgEdges = int(round(simpleGraph.sum() / numFinalNodes, 2))
    numInitNodes = numAvgEdges
    BA_graph = BA_Model(numInitNodes, numFinalNodes, numAvgEdges)
    return BA_graph


def BA_Model(numInitNodes, numFinalNodes, numAvgEdges, setSeed=0):
    # check input parameters
    if numInitNodes < numAvgEdges:
        raise ValueError("The number of initial connected nodes should be " 
        + "greater than the number of average edges to be added")

    # create an initial connected graph of numInitNodes nodes
    random.seed(setSeed)
    edgeProb = 0.5
    BA_graph = sparse.csc_matrix((numFinalNodes, numFinalNodes), dtype=np.int8)
    for nodeIdx in range(numInitNodes):
        sampleEdge = np.random.binomial(n=1, p=edgeProb, size=numInitNodes)
        existEdge = np.nonzero(sampleEdge)[0]
        existEdge = existEdge[np.nonzero(existEdge == nodeIdx)]
        BA_graph[nodeIdx, existEdge] = 1
    
    # Add new nodes one at a time
    for nodeIdx in range(numInitNodes, numFinalNodes):
        # calculate the probility of connecting to each existing node
        sumG = BA_graph.sum(axis=1)
        degree = np.squeeze(np.asarray(sumG))
        overallDegree = np.sum(degree)
        edgeProb = degree[:nodeIdx] / overallDegree

        # select the numAvgEdges most probable nodes to connect 
        mostProbableEdge = np.argsort((-1)*edgeProb)[:numAvgEdges]
        BA_graph[nodeIdx, mostProbableEdge] = 1
        BA_graph[mostProbableEdge, nodeIdx] = 1
    
    return BA_graph






 








    


