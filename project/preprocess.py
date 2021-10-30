from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

adjacency_matrix = np.loadtxt("datasets/adolescents/asd/Caltech_0051472.txt", dtype=int)
graph = nx.convert_matrix.from_numpy_matrix(adjacency_matrix)
