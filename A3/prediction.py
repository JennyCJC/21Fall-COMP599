from networkx.algorithms import node_classification
from helper import *
import networkx as nx

# node classification 
def predictNodeLabel(G, method, labelName='value'):
    if method == 'harmonic':
        predLabel = node_classification.harmonic_function(G, 
                        max_iter=10, label_name=labelName)
    elif method == 'consistency':
        predLabel =  node_classification.local_and_global_consistency(G, 
                        max_iter=10, label_name=labelName)
    return predLabel


def classify_real_classic_nodes(G, datasetName, labelName='value'):
    dropLabelPercentage = np.round(np.array([*range(95, 4, -20)]) * 0.01, 2)
    for method in ['harmonic', 'consistency']:
        accuracies = list()
        for dropPercentage in dropLabelPercentage:
            maskedG, dropLabel = dropLabels(G, dropPercentage, labelName)
            predLabel = predictNodeLabel(maskedG, method, labelName)
            accuracies.append(evaluateAccuracy(G, dropLabel, predLabel, labelName))

        plotAccuracy(dropLabelPercentage, accuracies, datasetName, method)
        printNodeClassifResult(datasetName, method, dropLabelPercentage, accuracies)


def classify_real_labelled_nodes(G, labels, testIdx):
    orderedLabels = np.nonzero(labels)[1]
    labelledG = addLabel2Nodes(G, testIdx, orderedLabels)
    predLabel_harmonic = predictNodeLabel(labelledG, 'harmonic') # predicted label for all nodes
    predlabel_consistency = predictNodeLabel(labelledG, 'consistency')



# link prediction
def linkPrediction_real_classic(G, ebunch=None):
    maskedG, droppedLinks = dropLinks(G)
    for method in ['jaccard']:
        predLinks = predict_links(maskedG, method, ebunch)



def predict_links(G, method, ebunch=None):
    if method == 'jaccard':
        predLinks = nx.jaccard_coefficient(G, ebunch=ebunch)



#def linkPrediction_real_labelled():












