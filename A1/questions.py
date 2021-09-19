import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy import sparse
from helpers import *
import math

def networkPatterns(simpleGraph):
    #degreeDistribution(simpleGraph)         #1A
    #clusterCoefDistribution(simpleGraph)    #1B
    #shortestPathDistribution(simpleGraph)   #1C
    #connectivity(simpleGraph)               #1D
    #eigenvalueDistribution(simpleGraph)     #1E
    degreeCorrelation(simpleGraph)          #1F
    #degreeClusterCoefRelation(simpleGraph)  #1G



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
    plt.xlabel('Degrees', fontsize=12.5)
    plt.ylabel('Frequency', fontsize=12.5)
    plt.show()
    
    #return slope
    print('Slope for the degree distribution is: ' + str(coefficients[0]))


def clusterCoefDistribution(simpleGraph):
    # calculate degree 
    sumG = simpleGraph.sum(axis=1)
    degree = np.squeeze(np.asarray(sumG))

    # calculate local clustering coefficient & frequency
    A2 = simpleGraph * simpleGraph
    A3 = A2 * simpleGraph
    numClosedPath = np.diagonal(A3.toarray())
    numPath = degree * (degree-1)
    finiteIdx = np.nonzero(numPath)
    finiteC = numClosedPath[finiteIdx] / numPath[finiteIdx]
    uniqueC, counts = np.unique(finiteC, return_counts=True)
    freq = np.cumsum(counts) / sum(counts)

    # compute and report the global clustering coefficient
    globalC = np.trace(A3.toarray()) / (np.sum(A2.toarray())-np.trace(A2.toarray()))
    avgC = np.sum(finiteC) / np.shape(simpleGraph)[0]
    print("Global clustering coefficient: " + str(globalC))
    print("Average clustering coefficient calculated by taking the mean:  " + str(avgC))

    # show the distribution in a plot
    plotGraph(uniqueC, freq, 'Clustering coefficient', 'Cumulative probability', 'Clustering coefficient distribution', 'line')



def shortestPathDistribution(simpleGraph):
    # calculate shortest distances & the corresponding frequencies
    distance = sparse.csgraph.shortest_path(simpleGraph)
    uniqueDistance, counts = np.unique(distance, return_counts=True)
    finiteIdx = np.isfinite(uniqueDistance)
    uniqueDistance = uniqueDistance[finiteIdx]
    counts = counts[finiteIdx]
    cumFreq = np.cumsum(counts) / sum(counts)
    freq = counts / sum(counts)

     #show shortest distance distribution with number of nodes
    plotGraph(uniqueDistance, counts, 'Shortest Distance', 'Number of Nodes', 'Shortest path distribution', 'line')

    #show the probability density in a plot
    plotGraph(uniqueDistance, freq, 'Shortest Distance', 'Probability density', 'Shortest path distribution (PDF)', 'line')

    #show the cumulative probability in a plot
    plotGraph(uniqueDistance, cumFreq, 'Shortest Distance', 'Cumulative probability', 'Shortest path distribution (CDF)', 'line')



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
    laplacian = laplacian.todense()
    # eval_max = sparse.linalg.eigs(laplacian, k=100, which='LM')

    #faster way to calculate eval_min
    eval_min = scipy.linalg.eigvalsh(laplacian, eigvals=(0, 2000))
    uniqueE, counts = np.unique(eval_min, return_counts=True)
    nonZero = uniqueE[np.nonzero(uniqueE)]
    print('Spectral gap is: ' + str(min(nonZero)))
    plotGraph(uniqueE, counts, 'Eigenvalues', 'Frequency', 'Eigenvalue Distribution', 'line')
 

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
    plotGraph(sourceDegree, destinationDegree, "Degree of source nodes", "Degree of destination nodes", "Degree correlation", 'scatter')
    # HAVEN'T BIN EDGES WITH LOW INTENSITY TO CAPTURE REGIONS WITH HIGH DENSITY



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
    plotGraph(degree, finiteC, "Degree of source nodes", "Degree of destination nodes", "Degree correlation", 'scatter')



def syntheticGraph(simpleGraph, model):
    numFinalNodes = np.shape(simpleGraph)[0]
    numAvgEdges = int(round(simpleGraph.sum() / numFinalNodes, 2))
    numInitNodes = numAvgEdges # HYPERPARAMETER
    graph = networkModel(model, numInitNodes, numFinalNodes, numAvgEdges, model)
    return graph



def networkModel(model, numInitNodes, numFinalNodes, numAvgEdges, setSeed=0):
    # check input parameters
    if numInitNodes < numAvgEdges:
        raise ValueError("The number of initial connected nodes should be " 
        + "greater than the number of average edges to be added")

    # create an initial connected graph of numInitNodes nodes
    # where each newly added node forms exactly one edge with an existing node
    syntheticGraph = createInitialGraph(numInitNodes, numFinalNodes, setSeed=0)
    
    # Add new nodes one at a time
    for nodeIdx in range(numInitNodes, numFinalNodes):
        if model=='BA' or model=='reverseBA':
            # calculate the probility of connecting to each existing node
            sumG = syntheticGraph.sum(axis=1)
            degree = np.squeeze(np.asarray(sumG))
            overallDegree = np.sum(degree)
            edgeProb = degree[:nodeIdx] / overallDegree

            if model == 'BA':
                # select the numAvgEdges nodes based on preferential attachment to form edges
                connectEdge = np.random.choice(range(nodeIdx), size=numAvgEdges, p=edgeProb)
                # print(connectEdge)
            elif model == 'reverseBA':
                # select the numAvgEdges based on the inverse of preferential attachment to form edges
                inverseEdgeProb=findInverseEdgeProb(edgeProb)
                connectEdge = np.random.choice(range(nodeIdx), size=numAvgEdges, p=inverseEdgeProb)  
        elif model == "indepAttachment":
            # randomly select m nodes to form edges
            connectEdge = np.random.choice(range(nodeIdx), size=numAvgEdges)

        syntheticGraph[nodeIdx, connectEdge] = 1
        syntheticGraph[connectEdge, nodeIdx] = 1

    return syntheticGraph

def createInitialGraph(numInitNodes, numFinalNodes, setSeed=0):
    random.seed(setSeed)
    syntheticGraph = sparse.csc_matrix((numFinalNodes, numFinalNodes), dtype=np.int8)
    for nodeIdx in range(1, numInitNodes):
        targetRange = range(nodeIdx)
        np.random.shuffle(targetRange)
        sampleEdge = targetRange[0]
        syntheticGraph[nodeIdx, sampleEdge] = 1
        syntheticGraph[sampleEdge, nodeIdx] = 1

    return syntheticGraph


def findInverseEdgeProb(edgeProb):
    sortIndex = np.argsort(edgeProb)
    nodeNum = len(sortIndex)
    sortInverse = nodeNum-sortIndex-1
    inverseEdgeProb = []
    for i in range(nodeNum):
        for j in range(nodeNum):
            if sortInverse[i] == sortIndex[j]:
                inverseEdgeProb.append(edgeProb[j])
                break
    return inverseEdgeProb

 








    


