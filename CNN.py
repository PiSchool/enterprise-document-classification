#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: robi
"""

def CNN_create(Nr,Nc, mod='AlexNet'):
    # Create CNN AlexNet for comparison (version of Tensmeyer paper, Aug 2017)
    from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
    from keras.models import Sequential
    
    model= Sequential()
    
    # Calculate
    
    # First conv layer
    # CONV
    model.add( Conv2D(96,(11,11),strides=(4,4),
                      padding='same',activation='relu',
                      input_shape=(Nr,Nc,1)) )
    # MAXP overlap
    model.add( MaxPooling2D(pool_size=(3,3),strides=2) )
    
    # Second conv layer
    # CONV
    model.add( Conv2D(256,(5,5),padding='same',activation='relu') )
    # MAXP overlap
    model.add( MaxPooling2D(pool_size=(3,3),strides=2) )
    
    # Third conv layer
    # CONV
    model.add( Conv2D(384,(3,3),padding='same',activation='relu') )
    # Fourth conv layer
    # CONV
    model.add( Conv2D(384,(3,3),padding='same',activation='relu') )
    
    # Fifth conv layer
    # CONV
    model.add( Conv2D(256,(3,3),padding='same',activation='relu') )
    # MAXP overlap
    model.add( MaxPooling2D(pool_size=(3,3),strides=2) )
    
    # First FC layer
    model.add( Flatten() )
    model.add( Dense(64, activation='relu') )
    # Second FC layer
    model.add( Dense(64, activation='relu') )
    # Predictions
    model.add( Dense(16, activation='softmax') )
    
    # Model creation
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model