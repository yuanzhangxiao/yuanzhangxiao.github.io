#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 04:25:59 2017

@author: yuanzhangxiao
"""

from sklearn.datasets import fetch_mldata
import numpy as np
    
def load_mnist(binary_digits, number_training):

    # fetch the MNIST data
    mnist = fetch_mldata('MNIST original', transpose_data=False)
    
    print(mnist.data.shape)

    # get training images and labels
    train_image = mnist.data[ :, np.arange(0,60000) ]
    train_label = mnist.target[ np.arange(0,60000) ]

    # get test images and labels
    test_image  = mnist.data[ :, np.arange(60000,70000) ]
    test_label  = mnist.target[ np.arange(60000,70000) ]

    # convert the training and test images from integers into real numbers
    train_image = np.asarray( train_image, dtype=np.float64 )
    test_image = np.asarray( test_image, dtype=np.float64 )

    # normalize the pixels from [0,255] to [0,1]
    train_image = train_image/ 255.0
    test_image  = test_image / 255.0

    # chech if we extract digits 0 and 1 only  
    if binary_digits == True:
        train_image = np.hstack( (train_image[ :, train_label==0 ] , train_image[ :, train_label==1 ]) )
        train_label = np.vstack( ( train_label[ train_label==0 ].reshape(np.sum(train_label==0),1) , train_label[ train_label==1 ].reshape(np.sum(train_label==1),1) ) )
        test_image = np.hstack( (test_image[ :, test_label==0 ] , test_image[ :, test_label==1 ]) )
        test_label = np.vstack( ( test_label[ test_label==0 ].reshape(np.sum(test_label==0),1) , test_label[ test_label==1 ].reshape(np.sum(test_label==1),1) ) )
    
    # randomly shuffle the data
    shuffle_index_train = np.arange( train_image.shape[1] )
    np.random.shuffle( shuffle_index_train )
    train_image = train_image[:, shuffle_index_train]
    train_label = train_label[shuffle_index_train]
    shuffle_index_test = np.arange( test_image.shape[1] )
    np.random.shuffle( shuffle_index_test )
    test_image = test_image[:, shuffle_index_test]
    test_label = test_label[shuffle_index_test]
            
    # take the first (number_training) training examples        
    train_image = train_image[:, np.arange(number_training)]
    train_label = train_label[np.arange(number_training)]        
    
    # standardize the data so that each pixel has zero mean and unit variance        
    mean = np.mean( train_image, axis = 1 ).reshape(784,1)
    std  = np.std( train_image, axis = 1 ).reshape(784,1)        
    train_image = train_image - mean;
    train_image = train_image / (std+.1)
    test_image = test_image - mean
    test_image = test_image / (std+.1)
    
    # make the labels row vectors
    train_label = np.transpose(train_label)
    test_label  = np.transpose(test_label)
    
    return train_image, train_label, test_image, test_label