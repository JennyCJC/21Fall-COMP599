import numpy as np
import math
import matplotlib.pyplot as plt

#count how many times an element occurs in an array
def count(elem, array):
    ct = 0
    for element in array:
        if elem == element:
            ct+=1
    return ct

#calculate the frequency of numbers in the set
def freq(x):
    freqs = [[value, count(value, x)] for value in set(x)] 
    return freqs

#bin size would be reduced if no data lies in the bin
def logBinning(binSize, degreeData):
    if binSize >= len(degreeData):
        raise ValueError('Your bin size should be smaller than the data size.')

    if binSize < 3:
        raise ValueError('Your bin size should be greater than 2 to allow meaningful bins')

    binRange = np.logspace(0, math.log(max(degreeData[:, 0]), 10), binSize-2)

    binnedData = []
    for i in range(0, binSize):
        binnedData.append([])

    for data in degreeData:
        prevBin = binRange[0]
        mark = False
        for i in range(0, binSize-3):
            if binRange[i] <= data[0] < binRange[i+1]:
                binnedData[i+1].append(data)
                mark = True
                break
        if not mark:
            if data[0] < binRange[0]:
                binnedData[0].append(data)
            else:
                binnedData[binSize-1].append(data)

    averagePoint = []
    for row in binnedData:
        averageX = 0
        averageY = 0
        num = 0
        if len(row) > 0:
            for data in row:
                averageX += data[0]
                averageY += data[1]
                num+=1
            averageX /= num
            averageY /= num
            averagePoint.append([averageX, averageY])

    return np.array(averagePoint)
            
def plotLine(xData, yData, xLabel, yLabel, title):
    plt.plot(xData, yData, linewidth=2.5)
    plt.title(title, fontsize=14)
    plt.xlim(min(xData), max(xData))
    plt.ylim(min(yData), max(yData))
    plt.xlabel(xLabel, fontsize=12.5)
    plt.ylabel(yLabel, fontsize=12.5)
    plt.show()