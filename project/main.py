import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import charikar as ch
from helper import *
from preprocess import *
from classify import *
from LPAlgorithms import findMinimal
from gurobipy import * 

#TODO: handle input I/O
k = 10
alpha = 0
algo = 1

#alpha is the threshold for the overlapping.
def densestSubgraph (dataset, subgraphNum=3, alpha=0.8, algo=1):
    g = loadGraph(dataset)
    nodeCt = len(g.nodes())
    i = 1
    toMaximalDensity = 0.0
    begginning = time.time
    timeCt = 0.0
    topSubgraphs = []

    while(len(g.edges()) > 0 and i <= subgraphNum):
        start = time.time
        print('Top ' + str(i) + ' of ' + str(k) + ' started...')
        #current is current densest subgraph
        current = set()

        if algo == 0:
            # current = approxChar
            print('not implemented yet')
        elif algo == 1:
            current = findMinimal(g)
        else:
            current, best_avg = ch.charikarDicts(g)
        
        currentSize = len(current)
        dataName = dataset+'alpha'+str(alpha)+'alg'+str(algo)
        topSubgraphs.append(printTopINodesSubgraph(dataName, i, current, False))

        g, m = removeWeakConnections(g, current, alpha)
        i+=1
    return topSubgraphs

    
def main():

    #diffG = make_difference_graph("datasets/children")
    #A = nx.adjacency_matrix(diffG)
    #np.savetxt("datasets/children/diffGraph", A.todense())
    #topSubgraphs = densestSubgraph("datasets/children/diffGraph", alpha=0.2)
    #nx.draw(topSubgraphs[0][1])
    #plt.show()

    topSubgraphs = densestSubgraph("datasets/adolescents/diffGraph", subgraphNum=5, alpha=0.9, algo=2)
    svm_classifier("male", topSubgraphs, method="graph2vec")

    #edgelist_files("datasets/male/td")


if __name__ == "__main__":
    main()