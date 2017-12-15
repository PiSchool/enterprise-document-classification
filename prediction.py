# -*- coding: utf-8 -*-
"""
Prediction/Classification Module:

Methods available:
    - Kmeans
    - ConvolutionalNN
    - K-NN
    
Used as "forward models" set with parameters learned at the previous step.		

INPUT: Test set - raw data
OUTPUT: Predicted class

@author: robi
"""

def test_pred(ret,img):
    """
    Prediction routine - very simple
    """
    import imagehash
    from PIL import Image
    from collections import OrderedDict
    
    # Create distance vector
    distance={}
    
    # convert img to feature vec
    hashh= imagehash.phash( Image.fromarray( img ) )
    
    # Calculate distances from classes represented by feature templates
    for c in ret:
        distance[c]= hashh-ret[c]
    
    dd= OrderedDict(sorted(distance.items(), key=lambda t: t[1]))
    
    return dd.keys()[0]