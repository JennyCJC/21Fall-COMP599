from scipy.optimize import linprog
import networkx as nx
from charikar import charikarDicts
import random
def getDensity(G):
    return len(G.edges)/len(G.nodes)

def basicLP(G):
    n = len(G.nodes)
    m = len(G.edges)

    c = [(-1 if i < m else 0) for i in range(n+m)]
    A = []

    for i in range(len(G.edges)):
        e = list(G.edges)[i]
        A1 = [0 for i in range(n + m)]
        A2 = [0 for i in range(n + m)]
        A1[i] = 1
        A1[e[0]+m] = -1

        A2[i] = 1
        A2[e[1]+m] = -1

        A.append(A1)
        A.append(A2)

    A.append([(1 if i >= m else 0) for i in range(n+m)])

    b = [0 for i in range(len(A)-1)]
    b.append(1)

    bounds = [(0, None) for i in range(n+m)]

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

    return [round(x, 6) for x in res.x]

def fastLP(G, basicLP_results):
    nodeVars = basicLP_results[len(G.edges):]
    inducedSubgraphNodes = [i for i in range(len(nodeVars)) if not nodeVars[i] == 0.0 ]

    return nx.convert_node_labels_to_integers(G.subgraph(inducedSubgraphNodes))

def tryRemove(G, v):
    initDens = getDensity(G)
    G = G.copy()
    G.remove_node(v)
    G = nx.convert_node_labels_to_integers(G)
    G = fastLP(G, basicLP(G))

    return G if getDensity(G) >= initDens else None

def tryEnchance(G, v, pmax):
    n = len(G.nodes)
    m = len(G.edges)

    c = [(-1 if i == m+v else 0) for i in range(n + m)]
    A = []

    for i in range(len(G.edges)):
        e = list(G.edges)[i]
        A1 = [0 for i in range(n + m)]
        A2 = [0 for i in range(n + m)]
        A1[i] = 1
        A1[e[0] + m] = -1

        A2[i] = 1
        A2[e[1] + m] = -1

        A.append(A1)
        A.append(A2)

    A.append([(1 if i >= m else 0) for i in range(n + m)])

    A.append([(-1 if i < m else 0) for i in range(n + m)])
    A.append([(1 if i < m else 0) for i in range(n + m)])

    b = [0 for i in range(len(A) - 3)]
    b.append(1)
    b.append(-pmax)
    b.append(pmax)

    bounds = [(0, None) for i in range(n + m)]

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

    if res.success:
        return fastLP(G, [round(x, 6) for x in res.x])
    else:
        return None

def findMinimal(G):
    G, avg_degree = charikarDicts(G)
    G = nx.convert_node_labels_to_integers(G)
    p_apx = getDensity(G)

    for node, deg in G.degree():
        if deg < p_apx:
            G.remove_node(node)
    G = nx.convert_node_labels_to_integers(G)

    G_ = fastLP(G, basicLP(G))
    p_max = getDensity(G)

    while(True):
        v = random.choice(list(G_.nodes))
        H1 = tryRemove(G_, v)
        H2 = tryEnchance(G_, v, p_max)

        if H1 is None:

            return H2
        elif H2 is None:
            G_ = H1
        else:
            G_ = H1 if len(H1.nodes) <= len(H2.nodes) else H2

