# -*- coding: utf-8 -*-
"""
Preproc: This module contains preprocessing routines like binarization, 
        adaptive thresholding, histogram equalization etc 

@author: robi
"""
import cv2
import numpy as np
import random

# Mapping of classes meaning
C={'letter': 0,
   'form': 1,
   'email': 2,
   'handwritten': 3,
   'advertisement': 4,
   'scientific report': 5,
   'scientific publication': 6,
   'specification': 7,
   'file folder': 8,
   'news article': 9,
   'budget': 10,
   'invoice': 11,
   'presentation': 12,
   'questionnaire': 13,
   'resume': 14,
   'memo': 15}

def create_data(df_train,df_test,df_val):
    """
    Dataset dictionary creation
    """
    # Create a dictionary that, for each class, contains all the files of the training
    # set labeled as pertaining to that class
    dataset= {'train': {},
              'test': {},
              'val': {}
              }
              
    K= len(C)
    nitems= np.zeros((3,K))
    for c in C:
        idx= C[c]
        if c not in dataset['train']: 
            dataset['train'][c]=[]
            dataset['test'][c]=[]
            dataset['val'][c]=[]
        dataset['train'][c]= df_train.query('Class==@idx')
        dataset['test'][c]= df_test.query('Class==@idx')
        dataset['val'][c]= df_val.query('Class==@idx')
        
        nitems[0,idx]= len(dataset['train'][c])
        nitems[1,idx]= len(dataset['test'][c])
        nitems[2,idx]= len(dataset['val'][c])
    
    return dataset, nitems
    
def create_sample(dataset, nitems, dataset_path, Nr, Nc, label= 'train',
            Nbatch= 100):
    """
    Generate actual image or text sample from data filenames
    """
    lb={
        'train': 0,
        'test': 1,
        'val': 2,
        }
    # label: defines training, test or validation set
    # Nbatch: defines the # samples
    # Nr: number of row pixels
    # Nc: number of columns pixels
    Cc= len(dataset[label].keys())
    
    # Create a dataset in matrix format
    dset=np.zeros( (Nbatch*Cc,Nr*Nc), dtype=np.uint8 )
    y=np.zeros( (Nbatch*Cc,1), dtype=np.uint8 )
    
    ik=0
    for c in dataset[label]:
        
        for kk in range(0,Nbatch):
            # Pick a random image from c class
            i= random.randint(0,nitems[lb[label],C[c]]-1)
            # Preprocess the image, resize (512x512) and save to the set
            img= prepare( cv2.imread(dataset_path+"images/"+dataset[label][c].values[i,0],0), Nr=Nr,Nc=Nc )
            dset[ik,:]= img.flatten()
            y[ik]= C[c]
            ik+=1
    
    return dset, y

def prepare(img, Nr=512, Nc=512):
    """
    Prepare the image for further preprocessing.
    Preparation steps:
        - Histogram equalization
        - Normalization
        - Resizing to (512x512) with cubic spline interpolation
    """

    # Thresholding
    img_tr, th= thresholding(img)    
    
    # Convert to float and rescale img
    img= np.double(img)/np.max(np.double(img))
    
    # Resizing with nearest-neighbour interpolation to retain amplitude dynamic
    img_n= cv2.resize(img_tr, (Nr,Nc), interpolation=cv2.INTER_AREA)
    
    return img_n

def thresholding(img,**kwargs):
    # Dictionary of defaults parameters
    kwargs={
            'thresh': 127,
            'min': 0,
            'max': 255,
            'type': 'otsu',
            'at_blocksize': 11,
            'at_constant': 0,
    }
    kwargs.update(**kwargs)
    
        
    if kwargs['type']=='global':
        # global thresholding
        ret,th = cv2.threshold(img,kwargs['thresh'],kwargs['max'],cv2.THRESH_BINARY)
    elif kwargs['type']=='otsu':
        # Otsu's thresholding
        ret,th = cv2.threshold(img,kwargs['min'],kwargs['max'],cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif kwargs['type']=='adaptive':
        # Adaptive thresholding with gaussian weighting of neighbourhood
        # Parameters: blocksize and arbitrary constant to be subtracted from result
        th = cv2.adaptiveThreshold(img,kwargs['max'],cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY,kwargs['at_blocksize'],
                                    kwargs['at_constant'])
    
    return img, th