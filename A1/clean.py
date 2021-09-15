import numpy as np
from scipy import sparse


#remove self-loops
def removeSelfLoop(edgeList):
    toRemove = []
    for i in range(0, len(edgeList)):
        edge = edgeList[i]
        if edge[0] == edge[1]:
            toRemove.append(i)
    return np.delete(edgeList, toRemove, axis=0)

#generate crs matrix
def generateCRS(edgeList):
    row = edgeList[:, 0]
    col = edgeList[:, 1]
    sizeOfMatrix = len(row)
    data = [1]*sizeOfMatrix
    return sparse.csr_matrix((data, (row, col)), shape=(sizeOfMatrix, sizeOfMatrix))

#make the graph undirected
def makeSymmetric(edgeList):
    A = generateCRS(edgeList)
    T = A.transpose()
    result = A.maximum(T)

    # check if A and T equal to result
    # if sum is 0 then equal
    # print(np.sum(T!=result))
    # print(np.sum(A!=result))

    return result

#starting point, data cleaning finished
def generateSimpleGraph(edgeList):
    l1 = removeSelfLoop(edgeList)
    l2 = makeSymmetric(l1)
    return l2

