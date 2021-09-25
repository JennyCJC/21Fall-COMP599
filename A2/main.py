import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from clean import * 
from centrality_Q1 import *
from community_Q2 import *
from helper import *


#Raw data
emailsList = np.loadtxt("email-Enron/email-Enron.txt", dtype=int)
# print(np.delete(emailsList, 2, 1))
# print(emailsList)
# print([emailsList[:, 0],emailsList[:, 1]])
# print(createInitialGraph(emails))
emailGraph = generateCSC(createUniqueEdges(np.delete(emailsList, 2, 1)))

strikeGraph = nx.read_gml('real-classic/strike.gml')
karateGraph = nx.read_gml('real-classic/karate.gml')
polbooksGraph = nx.read_gml('real-classic/polbooks.gml')
#polblogsGraph = nx.read_gml('real-classic/polblogs.gml')
#footballGraph = nx.read_gml('real-classic/football.gml')

coraDict = np.load('real-node-label/ind.cora.graph', allow_pickle=True)
coraGraph = nx.Graph(dict(coraDict))
citeseerDict = np.load('real-node-label/ind.citeseer.graph', allow_pickle=True)
citeseerGraph = nx.Graph(dict(citeseerDict))
pubmedDict = np.load('real-node-label/ind.pubmed.graph', allow_pickle=True)
pubmedGraph = nx.Graph(dict(pubmedDict))

#########


#Q1: Rank people based on centrality (Enron email dataset)
# mostImportantNodes(emailGraph)

#Q2: Community detection on graphsß
#real-classic datasets
# communityDetection(strikeGraph)
# communityDetection(karateGraph)
# communityDetection(polbooksGraph, plotLabel=False) 
# communityDetection(polblogsGraph)  # DUPLICATED EDGE ERROR? 
# communityDetection(footballGraph)  # DUPLICATED EDGE ERROR?

#real-node-label
communityDetection(coraGraph, plotLabel=False)