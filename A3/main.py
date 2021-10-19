import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from clean import * 
from helper import *
from prediction import *

#Processing data

# strikeGraph = nx.read_gml('real-classic/strike.gml')
# karateGraph = nx.read_gml('real-classic/karate.gml')
# polbooksGraph = nx.read_gml('real-classic/polbooks.gml')
# polblogsGraph = nx.read_gml('real-classic/polblogs.gml')
# footballGraph = nx.read_gml('real-classic/football.gml')

# coraDict = np.load('real-node-label/ind.cora.graph', allow_pickle=True)
# coraGraph = nx.Graph(dict(coraDict))
citeseerDict = np.load('real-node-label/ind.citeseer.graph', allow_pickle=True)
citeseerGraph = nx.Graph(dict(citeseerDict))
# pubmedDict = np.load('real-node-label/ind.pubmed.graph', allow_pickle=True)
# pubmedGraph = nx.Graph(dict(pubmedDict))

# coraLabel = np.load('real-node-label/ind.cora.ally', allow_pickle=True)
# coraTestIndex = np.loadtxt('real-node-label/ind.cora.test.index')
# coraTestLabel = np.load('real-node-label/ind.cora.ty', allow_pickle=True)

citeseerLabel = np.load('real-node-label/ind.citeseer.ally', allow_pickle=True)
citeseerTestIndex = np.loadtxt('real-node-label/ind.citeseer.test.index')
citeseerTestLabel = np.load('real-node-label/ind.citeseer.ty', allow_pickle=True)

# pubmedLabel = np.load('real-node-label/ind.pubmed.ally', allow_pickle=True)
# pubmedTestIndex = np.loadtxt('real-node-label/ind.pubmed.test.index')
# pubmedTestLabel = np.load('real-node-label/ind.pubmed.ty', allow_pickle=True)
#########


#Node classification
# classify_real_classic_nodes(strikeGraph, 'Strike dataset')
# classify_real_classic_nodes(karateGraph, 'Karate dataset')
# classify_real_classic_nodes(polbooksGraph, 'Polbooks dataset')
# classify_real_classic_nodes(polblogsGraph, 'Polblogs dataset')
# classify_real_classic_nodes(footballGraph, 'Football dataset')

# classify_real_labelled_nodes(coraGraph, coraLabel, coraTestIndex, coraTestLabel)
classify_real_labelled_nodes(citeseerGraph, citeseerLabel, citeseerTestIndex, citeseerTestLabel)
# classify_real_labelled_nodes(pubmedGraph, pubmedLabel, pubmedTestIndex, pubmedTestLabel)


#Link prediction
# linkPrediction_real_classic(strikeGraph)
# linkPrediction_real_classic(karateGraph)
# linkPrediction_real_classic(polbooksGraph) 
# linkPrediction_real_classic(multiGraphToSimpleGraph(polblogsGraph))    #multigraph
# linkPrediction_real_classic(footballGraph)

# linkPrediction_real_classic(coraGraph)
# linkPrediction_real_classic(citeseerGraph)
# linkPrediction_real_classic(pubmedGraph, 0.1)