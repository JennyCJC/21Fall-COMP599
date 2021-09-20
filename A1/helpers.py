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
def binning(binSize, degreeData, type=None):
    if binSize >= len(degreeData):
        raise ValueError('Your bin size should be smaller than the data size.')

    if binSize < 3:
        raise ValueError('Your bin size should be greater than 2 to allow meaningful bins')

    if type == 'loglog':
        binRange = np.logspace(0, math.log(max(degreeData[:, 0]), 10), binSize-2)
    elif type == None:
        binRange = np.linspace(0, max(degreeData[:, 0]), num=binSize-2)

    binnedData = []
    for i in range(0, binSize):
        binnedData.append([])

    for data in degreeData:
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
            
def plotGraph(xData, yData, xLabel, yLabel, title, type, scale='normal'):
    if scale == 'normal':
        if type == 'line':
            plt.plot(xData, yData, linewidth=2.5)
        elif type == 'scatter':
            plt.scatter(xData, yData)
    elif scale == 'loglog':
        if type == 'line':
            plt.loglog(xData, yData)
        elif type == 'scatter':
            plt.loglog(xData, yData, "o")
    plt.title(title, fontsize=14)
    plt.xlim(min(xData), max(xData)+1)
    plt.ylim(min(yData), max(yData)+1)
    plt.xlabel(xLabel, fontsize=12.5)
    plt.ylabel(yLabel, fontsize=12.5)
    plt.show()
