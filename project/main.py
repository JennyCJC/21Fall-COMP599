import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import charikar as ch
from helper import *
from preprocess import *

#TODO: handle input I/O
k = 10
alpha = 0
algo = 1

#alpha is the threshold for the overlapping.
def densestSubgraph (dataset, subgraphNum=3, alpha=0.2, algo=2):
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
            print('not implemented yet')
            
            # current = filtering(g)
        else:
            current, best_avg = ch.charikarDicts(g)
        
        currentSize = len(current)
        topSubgraphs.append(printTopINodesSubgraph(dataset+'alpha'+str(alpha)+'alg'+str(algo), i, current, False))

        g, m = removeWeakConnections(g, current, alpha)
        i+=1
    return topSubgraphs

    
def main():

    #diffG = make_difference_graph("datasets/children")
    #A = nx.adjacency_matrix(diffG)
    #np.savetxt("datasets/children/diffGraph", A.todense())
    topSubgraphs = densestSubgraph("datasets/children/diffGraph", alpha=0.2)
    nx.draw(topSubgraphs[0][1])
    plt.show()



if __name__ == "__main__":
    main()