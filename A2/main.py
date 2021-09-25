import numpy as np
from clean import * 
from centrality import *

#Raw data
emailsList = np.loadtxt("email-Enron/email-Enron.txt", dtype=int)
# print(np.delete(emailsList, 2, 1))
# print(emailsList)
# print([emailsList[:, 0],emailsList[:, 1]])
# print(createInitialGraph(emails))
emailGraph = generateCSC(createUniqueEdges(np.delete(emailsList, 2, 1)))
#########


#Q1: Rank people based on centrality (Enron email dataset)
mostImportantNodes(emailGraph)