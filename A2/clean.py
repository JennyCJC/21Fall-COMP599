import numpy as np
from scipy import sparse

def createUniqueEdges (edgeList):   
    uniqueC, counts = np.unique(edgeList, return_counts=True, axis=0)
    return [uniqueC, counts]

def generateCSC(uniqueEdges):
    edgeList = uniqueEdges[0]
    counts = uniqueEdges[1]
    row = edgeList[:, 0]
    col = edgeList[:, 1]
    size = max(max(row), max(col))
    return sparse.csc_matrix((counts, (row, col)), shape=(size+1, size+1))

