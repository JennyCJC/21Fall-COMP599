import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import charikar as ch
from helper import *
from preprocess import *
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


def graphFeature(G, topSubgraphs, G_asd, G_td, feature):
    '''
    extract the feature vector for each brain sample
    '''
    G_asd = nx.convert_matrix.from_numpy_matrix(np.mean(G_asd,-1))  # summary graph
    G_td = nx.convert_matrix.from_numpy_matrix(np.mean(G_td,-1))    # summary graph
    G = nx.convert_matrix.from_numpy_matrix(G)
    k = len(topSubgraphs)
    featureVec = np.empty(k*2)
    for i_subG in range(k):
        contrastSubG= topSubgraphs[i_subG][1]
        subgraph = G.subgraph(contrastSubG.nodes())
        subgraph_asd = G_asd.subgraph(contrastSubG.nodes())
        subgraph_td = G_td.subgraph(contrastSubG.nodes())
        featureVec[(i_subG-1)*2] = subgraph.size()- subgraph_asd.size(weight="weight")
        featureVec[(i_subG-1)*2+1] = subgraph.size()-subgraph_td.size(weight="weight")
    return featureVec            

    
def constructDesignMatrix(path, topSubgraphs, feature='numEdges'):
    '''
    Random 80/20 train/test split
    Construct x_train, y_train, x_test, y_test
    We use label 1 to indicate an "ASD patient", label 0 to indicate a healthy person
    ''' 
    G_asd = load_graphs(path, 'asd')
    n_asd = np.shape(G_asd)[-1]
    G_td = load_graphs(path, 'td')
    n_td = np.shape(G_td)[-1]
    n_total = n_asd + n_td
    
    G = np.concatenate((G_asd, G_td), axis=2)
    labels = np.concatenate((np.ones(n_asd), np.zeros(n_td)), axis=0)
    featureMatrix = np.empty((n_total, len(topSubgraphs)*2))
    for i in range(n_total):
        featureMatrix[i,:] = graphFeature(G[:,:,i], topSubgraphs, G_asd, G_td, feature)
 
    return featureMatrix, labels
    
    
    
def svm_classifier(path, topSubgraphs, method="Contrast_SubG"):
    '''
    Train a support vector machine classifier
    Use k-fold cross validation to tune the hyperparameter C
    Report the test accuracy
    '''
    if method=="graph2vec":
        features, labels = load_graph2vec_features(path)
    else:
        features, labels = constructDesignMatrix(path, topSubgraphs, feature='numEdges')
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    
    # Hyperparameter tuning
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 1)
    grid.fit(X_train, y_train)
    grid_predictions = grid.predict(X_test)
    print(np.sum(grid_predictions == y_test)/np.shape(y_test)[0])
    
    

def decisionTree_classifier(path, topSubgraphs):
    '''
    Train a decision tree classifier
    Return the k-fold cross validation accuracy or test accuracy
    '''
    (features, labels) = constructDesignMatrix(path, topSubgraphs, feature='numEdges')
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    cross_val_score(clf, X_train, y_train, cv=10)
    print(confusion_matrix(y_test, clf.predict(X_test)))
    
    
'''
Other functions that visualize decision boundary
EXPLANIABILITY!
'''