import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import charikar as ch
from helper import *

#TODO: handle input I/O
k = 10
alpha = 0
algo = 1

#alpha is the threshold for the overlapping.
def densestSubgraph (dataset, subgraphNum=10, alpha=0.0, algo=2):
    g = loadGraph(dataset)
    nodeCt = len(g.nodes())
    i = 1
    toMaximalDensity = 0.0
    begginning = time.time
    timeCt = 0.0
    topSubgraphs = []

    while(len(g.edges()) > 0 and i <= subgraphNum):
        start = time.time
        print('Top' + str(i) + ' of ' + str(k) + ' started...')
        current = set()

        if algo == 0:
            # current = approxChar
            print('not implemented yet')
        elif algo == 1:
            print('not implemented yet')
            
            # current = filtering(g)
        else:
            current = ch.charikarDicts(g)
        
        currentSize = len(current)
        topSubgraphs.append(printTopINodesSubgraph(dataset+'alpha'+alpha+'alg'+algo, i, current, flase))

        m = removeWeakConnections(g, current, alpha)


    
def main():
    densestSubgraph("datasets/adolescents/asd/Caltech_0051472.txt")

if __name__ == "__main__":
    main()