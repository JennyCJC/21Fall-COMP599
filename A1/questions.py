import numpy as np
import matplotlib.pyplot as plt
from helpers import freq

def degreeDistribution(simpleGraph):
    #calculate degree and frequency
    sumG = simpleGraph.sum(axis=1)
    sumArray = np.squeeze(np.asarray(sumG))
    degreeData = np.array(freq(sumArray))

    #show data in plot
    plt.loglog(degreeData[:, 0],degreeData[:, 1])
    plt.show()


# ten = [10]
# powerDegree = [0.5, 1.0, 1.5, 2.0]
# powerFreq = [1, 2, 3, 4]

# np.polyfit(np.power(ten, powerDegree), np.power(ten, powerFreq), )
# print(np.power(ten, powerDegree))
# print(stats.relfreq(sumArray, numbins=100))

