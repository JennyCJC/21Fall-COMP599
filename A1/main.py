import numpy as np
from clean import generateSimpleGraph
from questions import *

#########
#Raw data
proteinEdgeList = np.loadtxt("networks/protein.edgelist.txt", dtype=int)
powergridEdgeList = np.loadtxt("networks/powergrid.edgelist.txt", dtype=int)
metabolicEdgeList = np.loadtxt("networks/metabolic.edgelist.txt", dtype=int)

#########
#Simple Graphs

proteinSimpleGraph = generateSimpleGraph(proteinEdgeList)
powergridSimpleGraph = generateSimpleGraph(powergridEdgeList)
metabolicSimpleGraph = generateSimpleGraph(metabolicEdgeList)


#########
#Execution

def main():
    # Question 1
    networkPatterns(proteinSimpleGraph)
    networkPatterns(powergridSimpleGraph)
    networkPatterns(metabolicSimpleGraph)

    # Question 3
    proteinSyntheticGraph1 = syntheticGraph(proteinSimpleGraph, "BA")
    networkPatterns(proteinSyntheticGraph1)
    proteinSyntheticGraph2 = syntheticGraph(proteinSimpleGraph, "reverseBA")
    networkPatterns(proteinSyntheticGraph2)

    proteinSyntheticGraph3 = syntheticGraph(proteinSimpleGraph, "indepAttachment")
    networkPatterns(proteinSyntheticGraph3)

    metabolicSyntheticGraph1 = syntheticGraph(metabolicSimpleGraph, "BA")
    networkPatterns(metabolicSyntheticGraph1)
    metabolicSyntheticGraph2 = syntheticGraph(metabolicSimpleGraph, "reverseBA")
    networkPatterns(metabolicSyntheticGraph2)
    metabolicSyntheticGraph3 = syntheticGraph(metabolicSimpleGraph, "indepAttachment")
    networkPatterns(metabolicSyntheticGraph3)

    powergridSyntheticGraph1 = syntheticGraph(powergridSimpleGraph, "BA")
    networkPatterns(powergridSyntheticGraph1)
    powergridSyntheticGraph2 = syntheticGraph(powergridSimpleGraph, "reverseBA")
    networkPatterns(powergridSyntheticGraph2)
    powergridSyntheticGraph3 = syntheticGraph(powergridSimpleGraph, "indepAttachment")
    networkPatterns(powergridSyntheticGraph3)
    


if __name__ == "__main__":
    main()