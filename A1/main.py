import numpy as np
from clean import generateSimpleGraph
from questions import *

#########
#Raw data
# proteinEdgeList = np.loadtxt("networks/protein.edgelist.txt", dtype=int)
# citationEdgeList = np.loadtxt("networks/citation.edgelist.txt", dtype=int)
# wwwEdgeList = np.loadtxt("networks/www.edgelist.txt", dtype=int)
# internetEdgeList = np.loadtxt("networks/internet.edgelist.txt", dtype=int)
# phonecallsEdgeList = np.loadtxt("networks/phonecalls.edgelist.txt", dtype=int)
powergridEdgeList = np.loadtxt("networks/powergrid.edgelist.txt", dtype=int)
# using protein to test for now, reading all the data in at once is sloooooowwwwwww


#########
#Simple Graphs and Execution of functions

# proteinSimpleGraph = generateSimpleGraph(proteinEdgeList)
# citationSimpleGraph = generateSimpleGraph(citationEdgeList)
# internetSimpleGraph = generateSimpleGraph(internetEdgeList)
# phonecallsSimpleGraph = generateSimpleGraph(phonecallsEdgeList)
powergridSimpleGraph = generateSimpleGraph(powergridEdgeList)


#########
#Execution

def main():
    # Question 1
    # networkPatterns(proteinSimpleGraph)
    # networkPatterns(phonecallsSimpleGraph)
    networkPatterns(powergridSimpleGraph)
    # Question 3
    # proteinBAGraph = create_BA_Graph(proteinSimpleGraph)
    # networkPatterns(proteinBAGraph)
    


if __name__ == "__main__":
    main()