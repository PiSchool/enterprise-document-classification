# -*- coding: utf-8 -*-
"""
Learning/Training Module:

Methods available:
    - Kmeans
    - ConvolutionalNN
    - K-NN
    
Initialized with random values or a-priori information if available; this step
has the objective of optimizing the parameters of the models supported in order
to maximize the classification accuracy, given the INPUT as the training set.		

INPUT: training set - set of (features, class) pairs
OUTPUT: Learned parameters set

@author: robi
"""
from sklearn import cluster
from sklearn import linear_model
from sklearn import ensemble

def learn(feat_set,y, method='LogReg'):
    """
    Training of different classifiers
    """
    
    if method=='mbK-means':
        clasf= cluster.MiniBatchKMeans(n_clusters=16, init='k-means++', max_iter=300, 
                              random_state=None).fit(feat_set)
    elif method=='LogReg':
        clasf= linear_model.LogisticRegression(penalty='l1', class_weight='balanced',
                                                       solver='saga', multi_class='multinomial',
                                                       warm_start='False',
                                                       max_iter= 100,
                                                       n_jobs=-1).fit(feat_set,y.ravel())
    elif method=='ParForest':
        clasf= ensemble.ExtraTreesClassifier(n_estimators=1000,
                                                     max_features=128,
                                                     n_jobs=-1,
                                                     random_state=0).fit(feat_set,y.ravel())
        
    
    return clasf