from networkx.algorithms import node_classification
from helper import *
import networkx as nx
from sklearn import metrics

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
    print(G.nodes(data=True))
    predLabel_harmonic = predictNodeLabel(labelledG, 'harmonic') # predicted label for all nodes
    predlabel_consistency = predictNodeLabel(labelledG, 'consistency')



# link prediction

def predict_links(G, droppedLinks, method, percentage=None):
    if percentage > 0:
        ebunchPred = randomSelection(percentage, droppedLinks)
        # ebunchNon = randomSelection(percentage, list(nx.classes.function.non_edges(G)))
        ebunchNon = findNonEdges(G, percentage)
        ebunch = ebunchPred + ebunchNon
    else:
        ebunchPred = droppedLinks
        ebunch = None

    if method == 'jaccard':
        preds = nx.jaccard_coefficient(G, ebunch)
        fpr, tpr = generateLinkFromCoefficient(preds, ebunchPred, 0.025)
        return fpr, tpr
    elif method == 'preferential attachment': 
        preds = nx.preferential_attachment(G, ebunch)
        fpr, tpr = generateLinkFromCoefficient(preds, ebunchPred, 0.025)
        return fpr, tpr


def linkPrediction_real_classic(G, percentage=None):
    for method in ['jaccard', 'preferential attachment']:
        fprList = []
        tprList = []
        for i in range(10):
            maskedG, droppedLinks = dropLinks(G)
            fpr, tpr = predict_links(maskedG, droppedLinks, method, percentage)
            fprList.append(fpr)
            tprList.append(tpr)
        fprList.sort()
        tprList.sort()
        print(metrics.auc(fprList, tprList))












