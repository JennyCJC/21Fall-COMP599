import networkx as nx
import numpy as np


def initiateLeaders(G, k, thresh, centrality="degree"):
    # Initiate k leaders selected from nodes in G, based on their centrality
    # and the constraint that leaders should not be too iClose to each other
    # Inputs:
    #    G: input networkx graph
    #    k: number of leaders to initiate
    #    thresh: the threshold for the iCloseness between leaders
    #    centrality: an optional input that specifies which centrality measure to use
    #                to define the importance of nodes
    # Output:
    #   leaderNodes: the label of the nodes corresponding to the k selected leaders

    leaderNodes = []
    if centrality=="degree":
        dictDegreeCentrality = nx.algorithms.degree_centrality(G)
        rankings = np.asarray(list(dictDegreeCentrality.values()))
        mostImportantIdx = np.argsort((-1) * rankings)
        leaderNodes.append(mostImportantIdx[0])   # add the most important node as the 1st leader

        for potentialLeader in mostImportantIdx[1:]:
            if len(leaderNodes) < k:
                tooClose = False
                for existingLeader in leaderNodes:
                    if getCloseness(G, (potentialLeader, existingLeader)) <= thresh:
                        tooClose = True
                        break
                if not tooClose:
                    leaderNodes.append(potentialLeader)
            else:   # if we already have k leaders, break the for loop
                break

    return leaderNodes

                
        
def getCloseness(G, nodePair):
    # Calculate and return the iCloseness between node v1 and node v2 in graph G
    # Inputs:
    #    G: a networkx graph
    #    nodePair: (v1, v2), a tuple storing the two nodes whose iCloseness will be returned
    # Outputs:
    #    iCloseness: the iCloseness score between node v1 and node v2

    v1, v2 = nodePair
    allCommonNeighbors = list(nx.classes.function.common_neighbors(G, v1, v2))

    iCloseness = 0
    for commonNeighbor in allCommonNeighbors:
        nsProduct = getNeighborScoring(commonNeighbor, v1) * getNeighborScoring(commonNeighbor, v2)
        iCloseness = iCloseness + nsProduct
    
    return iCloseness


def getNeighborScoring(G, u, v):
    # Calculate and return the Neighbors Scoring (NS) between node u and node v in graph G.
    # Inputs:
    #    G: a networkx graph
    #    u: label of one networkx node
    #    v: label of another networkx node
    # Outputs:
    #    NS: the Neighbors Scoring between node u and node v

    if nx.algorithms.shortest_paths.generic.has_path(G, u, v):
        if u in G.neighbors(v):     
            NS = 1      # ns_1(u,v)
        else: 
            NS = 0      # ns_1(u,v)
        exploredEdges = list(nx.algorithms.traversal.breadth_first_search.bfs_edges(G, 
                    source=v, depth_limit=1))




    # node u and v are disconnected
    else:  
        return 0

        

















