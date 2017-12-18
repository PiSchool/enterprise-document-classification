# -*- coding: utf-8 -*-
"""
Feature Extraction Module:

Methods available:
    - PHASH
    - Image tomography
    - T-SVD
    
		

INPUT: image matrix or set of image matrices pertaining to a class
OUTPUT: feature vector for that image or for the entire input image set

@author: robi
"""

def feat_extr(test_set, Nr,Nc, feat_type='phash', eigen=5, hash_size=64, nclass=16):
    """
    Feature extractor from batches of training data
    """
    import imagehash
    import numpy as np
    import sklearn.decomposition as deco 
    import binascii as ba
    import PIL.Image as Image
    
    # Supported feature extraction methods
    methods= ['phash','ahash','tsvd']
    
    # Initialize feature set
    samples= np.shape(test_set)[0]
    if feat_type=='phash' or feat_type=='ahash':
        feat_set= np.zeros( ( samples,int((hash_size**2)/4) ), dtype=np.uint8)
    elif feat_type=='tsvd':
        # Define dimension of TSVD feature set and reserve space
        Nb= samples/nclass
        maxrank= np.min(Nr*Nc,Nb)
        feat_set= np.zeros((nclass,maxrank*eigen), dtype=np.uint8)
    else:
        # Do nothing; ensemble learning option
        feat_set= test_set
        
    # Loop on classes
    if feat_type in methods:
        for ik in range(samples):
            # Phash case - convert in feature space
            if feat_type=='phash':
                    feat_set[ik,:]= np.fromstring( 
                            ba.hexlify(
                            bytearray.fromhex(
                            imagehash.phash_simple( 
                                    Image.frombytes('L',(Nr,Nc),np.reshape(test_set[ik,:],(Nr,Nc)) ), 
                                    hash_size=hash_size ))), dtype=np.uint8)
            elif feat_type=='ahash':
                    feat_set[ik,:]= np.fromstring( 
                            ba.hexlify(
                            bytearray.fromhex(
                            imagehash.average_hash( 
                                    Image.frombytes('L',(Nr,Nc),np.reshape(test_set[ik,:],(Nr,Nc)) ),
                                    hash_size=hash_size ))), dtype=np.uint8)
            # T-SVD case
            elif feat_type=='tsvd':
                # Create decomposition function
                svd= deco.TruncatedSVD(n_components=eigen, n_iter=7, random_state=42)
                # Create feature set
                for ik in range(nclass):
                    feat_set[ik,:]= np.reshape(svd.transform( 
                            test_set[ik:(ik+1)*Nb,:] ),(1,maxrank*eigen),order='F')
    
    return feat_set

def inv_test(test_set,**kwargs):
    """
    Test feature invariance to different type of "attack"
    INPUTS:
        test_set: dictionary of N images per class
        feat_type: features type selected [phash,t-svd]
        attack_type: attack type selected [rotation, scaling, affine]
    OUTPUTS:
        test_set: features added to each list
        attack_vector: attack vector use for comparison
    """
    import imagehash
    import cv2
    import numpy as np
    from PIL import Image
    import sklearn.decomposition as deco    
    
    kwargs={
            'feat_type': 'phash',
            'attack_type': 'rotation',
            'rot_span': 120,
            'rot_spacing': 10,
    }
    kwargs.update(**kwargs)
    
    if kwargs['attack_type']=='rotation':
        # For each image of each class
        rot_vec= np.linspace(-kwargs['rot_span']/2,
                                  kwargs['rot_span']/2-kwargs['rot_spacing'],
                                  kwargs['rot_spacing'] )
        for c in test_set:
            rows,cols = test_set[c][0].shape
            for ii in rot_vec:
                M = cv2.getRotationMatrix2D((cols/2,rows/2),ii,1)
                if kwargs['feat_type']=='phash':
                    hashh= imagehash.phash( Image.fromarray( cv2.warpAffine(test_set[c][0],M,(cols,rows)) ) )
                    test_set[c].append( hashh  )
                elif kwargs['feat_type']=='tsvd':
                    svd= deco.TruncatedSVD(n_components=5, n_iter=7, random_state=42)
                    test_set[c].append( svd.transform( cv2.warpAffine(test_set[c][0],M,(cols,rows)) ) )
        
        return test_set, rot_vec
    else:
        print("Method not available")
        return
        
def mc_hashd_matrix(test,rot):
    """
    This routine builds a phash distance matrix which measure the Hamming distance
    among the phashes measured:
        - Intra-class on a sampled attack space (e.g. sampled rotational space)
        - Inter-class among different phashed value of the attacked image
    """
    import numpy as np
    # Represent the distance between hashes of same image and different orientation
    KK= list(test.keys())
    dimgh= np.zeros( (rot.size*len(KK),rot.size*len(KK)) )
    i=0
    for c1 in test:
        for ii1 in range(rot.size):
            i2=0
            for c2 in test:
                for ii2 in range(rot.size):
                    dimgh[i*rot.size+ii1,i2*rot.size+ii2]= test[c1][1+ii1]-test[c2][1+ii2]
                
                i2+=1
        i+=1
        
    return dimgh, KK