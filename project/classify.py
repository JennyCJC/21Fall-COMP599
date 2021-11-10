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


def graphFeature(G, topSubgraphs, feature):
    '''
    extract the feature vector for each brain sample
    '''
    k = len(topSubgraphs)
    featureVec = np.empty((1,k))
    for i_subG in range(k):
        contrastSubG= topSubgraphs[i_subG][1]
        subgraph = G.subgraph(contrastSubG.nodes())
        if feature=='numEdges':
            featureVec[i_subG] = subgraph.number_of_edges()
    return featureVec            

    
def constructDesignMatrix(path, topSubgraphs, feature='numEdges'):
    '''
    Random 80/20 train/test split
    Construct x_train, y_train, x_test, y_test
    We use label 1 to indicate an "ASD patient", label 0 to indicate a healthy person
    ''' 
    G_asd = load_graphs(path, 'asd')
    n_asd = np.shape(G_asd, -1)
    G_td = load_graphs(path, 'td')
    n_td = np.shape(G_td, -1)
    n_total = n_asd + n_td
    
    G = np.stack((G_asd, G_td), axis=-1)
    labels = np.vstack((np.ones((n_asd, 1), np.zeros((n_td, 1)))))
    featureMatrix = np.empty((n_total, len(topSubgraphs)))
    for i in range(n_total):
        featureMatrix[i,:] = graphFeature(G[:,:,i], topSubgraphs, feature)
 
    return (featureMatrix, labels)
    
    
    
def svm_classifier(path, topSubgraphs):
    '''
    Train a support vector machine classifier
    Use k-fold cross validation to tune the hyperparameter C
    Report the test accuracy
    '''
    (features, labels) = constructDesignMatrix(path, topSubgraphs, feature='numEdges')
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    
    # Hyperparameter tuning
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, y_train)
    grid_predictions = grid.predict(X_test)
    print(confusion_matrix(y_test, grid_predictions))
    
    

def decisionTree_classifier(path, topSubgraphs):
    '''
    Train a decision tree classifier
    Return the k-fold cross validation accuracy or test accuracy
    '''
    (features, labels) = constructDesignMatrix(path, topSubgraphs, feature='numEdges')
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    
    clf = DecisionTreeClassifier(random_state=0)
    cross_val_score(clf, X_train, y_train, cv=10)
    print(confusion_matrix(y_test, clf.predict(X_test)))
    
    
'''
Other functions that visualize decision boundary
EXPLANIABILITY!
'''