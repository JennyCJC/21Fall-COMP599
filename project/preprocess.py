import os
from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def load_graphs(path, category):
    nNodes = 116
    if category=="asd":
        path = path + '/asd/'
    elif category=='td':
        path = path + '/td/'
        
    n = len(os.listdir(path))
    graphs = np.zeros((nNodes, nNodes, n))
    i = 0

    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            graphs[:,:,i] = np.loadtxt(path + file, dtype=float)
            i = i+1
        
    return graphs
    

def make_summary_graph(path, output='graph'):
    w_asd = load_graphs(path, 'asd')
    w_td = load_graphs(path, 'td')
   
    if output=='graph':
        G_asd = nx.convert_matrix.from_numpy_matrix(np.mean(w_asd,-1))
        G_td = nx.convert_matrix.from_numpy_matrix(np.mean(w_td,-1))
    elif output=='weight':
        G_asd = np.mean(w_asd,-1)
        G_td = np.mean(w_td,-1)
    return (G_asd, G_td)


def make_difference_graph(path, symmetry=True):
    (w_asd, w_td) = make_summary_graph(path, output='weight')
    return nx.convert_matrix.from_numpy_matrix(np.abs(w_asd - w_td))


def weighted_oqc(G, alpha=1/3.):
    '''
        input: a subgraph of the difference graph
        output: the optimal quasi-clique score
    '''
    W = sum(data['weight'] for u, v, data in G.edges_iter(data=True))
    return W - alpha * ((len(G) * (len(G) - 1)) / 2.)


def oqc(G, alpha=1/3.):
    return G.number_of_edges() - alpha * ((len(G) * (len(G) - 1)) / 2.)

