
import numpy as np
from clean import * 

#Raw data
emailsList = np.loadtxt("email-Enron/email-Enron.txt", dtype=int)
# print(np.delete(emailsList, 2, 1))
# print(emailsList)
# print([emailsList[:, 0],emailsList[:, 1]])
# print(createInitialGraph(emails))
generateCSC(createUniqueEdges(np.delete(emailsList, 2, 1)))
#########