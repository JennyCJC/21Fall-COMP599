
def convertFrozenset2Dict(partitionList):
    sets = [list(x) for x in partitionList]
    partition = {}
    for clusterIdx in range(len(sets)):
        for node in sets[clusterIdx]:
            partition[node] = clusterIdx
    return partition


def convertLabel2Dict(labels):    
    partition = {}
    for nodeIdx in range(1, len(labels)+1):
        partition[str(nodeIdx)] = labels[nodeIdx-1]
    print(partition)
    return partition