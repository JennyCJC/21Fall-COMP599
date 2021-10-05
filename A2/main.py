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

strikeGraph = nx.read_gml('real-classic/strike.gml')
karateGraph = nx.read_gml('real-classic/karate.gml')
polbooksGraph = nx.read_gml('real-classic/polbooks.gml')
polblogsGraph = nx.read_gml('real-classic/polblogs.gml')
footballGraph = nx.read_gml('real-classic/football.gml')

coraDict = np.load('real-node-label/ind.cora.graph', allow_pickle=True)
coraGraph = nx.Graph(dict(coraDict))
citeseerDict = np.load('real-node-label/ind.citeseer.graph', allow_pickle=True)
citeseerGraph = nx.Graph(dict(citeseerDict))
pubmedDict = np.load('real-node-label/ind.pubmed.graph', allow_pickle=True)
pubmedGraph = nx.Graph(dict(pubmedDict))

coraLabel = np.load('real-node-label/ind.cora.ally', allow_pickle=True)
coraTestIndex = np.loadtxt('real-node-label/ind.cora.test.index')
coraOrderedLabels = np.nonzero(coraLabel)[1]

citeseerLabel = np.load('real-node-label/ind.citeseer.ally', allow_pickle=True)
citeseerTestIndex = np.loadtxt('real-node-label/ind.citeseer.test.index')
citeseerOrderedLabels = np.nonzero(citeseerLabel)[1]
citeseerGraph = removeUnlabeledCiteseerNodes(citeseerGraph, citeseerTestIndex, np.nonzero(citeseerLabel)[0])

pubmedLabel = np.load('real-node-label/ind.pubmed.ally', allow_pickle=True)
pubmedTestIndex = np.loadtxt('real-node-label/ind.pubmed.test.index')
pubmedOrderedLabels = np.nonzero(pubmedLabel)[1]
#########


#Q1: Rank people based on centrality (Enron email dataset)
mostImportantNodes(emailGraph)

#Q2: Community detection on graphsß
#real-classic datasets
strikePredictions = communityDetection(strikeGraph)
karatePredictions = communityDetection(karateGraph)
polbooksPredictions = communityDetection(polbooksGraph, plotLabel=False) 
communityDetection(polblogsGraph, plotLabel=False)  # DUPLICATED EDGE ERROR? 
footballPredictions = communityDetection(footballGraph, plotLabel=False)  

#real-node-label
coraPredictions = communityDetection(coraGraph, plotLabel=False)
citeseerPredictions = communityDetection(citeseerGraph, plotLabel=False)
pubmedPredictions = communityDetection(pubmedGraph, plotLabel=False)

#LFR
LFRGraph = nx.generators.community.LFR_benchmark_graph(1000, 3, 1.5, 0.5, average_degree=5, min_community=20, seed=3)
LFRPredictions = communityDetection(LFRGraph, plotLabel=False)
LFRcommunities = getUniqueCommunities(LFRGraph)
LFRLableDict = getLabelDictFromCommunities(LFRcommunities)



# Performance
overallPerformance(karateGraph, karatePredictions)
overallPerformance(strikeGraph, strikePredictions)
overallPerformance(polbooksGraph, polbooksPredictions)
overallPerformance(footballGraph, footballPredictions)
overallPerformance(coraGraph, coraPredictions, removeUnlabeled=coraTestIndex, truthLabels=coraOrderedLabels)
overallPerformance(citeseerGraph, citeseerPredictions, removeUnlabeled=citeseerTestIndex, truthLabels=citeseerOrderedLabels)
overallPerformance(pubmedGraph, pubmedPredictions, removeUnlabeled=pubmedTestIndex, truthLabels=pubmedOrderedLabels)
overallPerformance(LFRGraph, LFRPredictions, truthLabels=getLabels(LFRLableDict))