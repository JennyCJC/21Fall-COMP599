import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from clean import * 
from centrality_Q1 import *
from community_Q2 import *
from helper import *


#Processing data

emailsList = np.loadtxt("email-Enron/email-Enron.txt", dtype=int)
emailGraph = generateCSC(createUniqueEdges(np.delete(emailsList, 2, 1)))

# strikeGraph = nx.read_gml('real-classic/strike.gml')
# karateGraph = nx.read_gml('real-classic/karate.gml')
# polbooksGraph = nx.read_gml('real-classic/polbooks.gml')
# polblogsGraph = nx.read_gml('real-classic/polblogs.gml')
# footballGraph = nx.read_gml('real-classic/football.gml')

coraDict = np.load('real-node-label/ind.cora.graph', allow_pickle=True)
coraGraph = nx.Graph(dict(coraDict))
# citeseerDict = np.load('real-node-label/ind.citeseer.graph', allow_pickle=True)
# citeseerGraph = nx.Graph(dict(citeseerDict))
# pubmedDict = np.load('real-node-label/ind.pubmed.graph', allow_pickle=True)
# pubmedGraph = nx.Graph(dict(pubmedDict))

#########


#Q1: Rank people based on centrality (Enron email dataset)
# mostImportantNodes(emailGraph)

#Q2: Community detection on graphsß
#real-classic datasets
# communityDetection(strikeGraph)
# karatePredictions = communityDetection(karateGraph)
# communityDetection(polbooksGraph, plotLabel=False) 
# communityDetection(polblogsGraph, plotLabel=False)  # DUPLICATED EDGE ERROR? 
# communityDetection(footballGraph, plotLabel=False)  

#real-node-label
communityDetection(coraGraph, plotLabel=False)
# communityDetection(citeseerGraph, plotLabel=False)
# communityDetection(pubmedGraph, plotLabel=False)

#LFR
# LFRGraph = nx.generators.community.LFR_benchmark_graph(1000, 3, 1.5, 0.5, average_degree=5, min_community=20, seed=3)
# communityDetection(LFRGraph, plotLabel=False)

#Performance
#overallPerformance(karateGraph, karatePredictions)
