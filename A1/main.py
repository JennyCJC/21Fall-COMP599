import numpy as np
from clean import generateSimpleGraph
from questions import degreeDistribution

#########
#Raw data
proteinEdgeList = np.loadtxt("networks/protein.edgelist.txt", dtype=int)
# citationEdgeList = np.loadtxt("networks/citation.edgelist.txt", dtype=int)
# wwwEdgeList = np.loadtxt("networks/www.edgelist.txt", dtype=int)
# using protein to test for now, reading all the data in at once is sloooooowwwwwww


#########
#Simple Graphs and Execution of functions

proteinSimpleGraph = generateSimpleGraph(proteinEdgeList)
#citationSimpleGraph = generateSimpleGraph(citationEdgeList)


#########
#Execution

def main():
    #1A
    degreeDistribution(proteinSimpleGraph)

if __name__ == "__main__":
    main()