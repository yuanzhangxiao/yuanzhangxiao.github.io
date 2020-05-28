#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 04:25:59 2017

@author: yuanzhangxiao
"""

# from sklearn.datasets import fetch_mldata
import numpy as np

"""
load the data:
use load_mnist to load the data containing digits 0 and 1
use load_mnist_5_6 to load the data containing digits 5 and 6
"""
# load the data containing digits 0 and 1
#binary_digits = True;
#number_training = 50;
#train_image, train_label, test_image, test_label = load_mnist(binary_digits=True, number_training = 100)

# load the data containing digits 5 and 6
train_image, train_label, test_image, test_label = load_mnist_5_6(number_training = 50)

# add row of 1s to the dataset to act as an intercept term
train_image = np.vstack( ( np.ones( ( 1, train_image.shape[1] ) ) , train_image ) )
test_image  = np.vstack( ( np.ones( ( 1, test_image.shape[1]  ) ) , test_image  ) )

# training set dimensions
m = train_image.shape[1];
n = train_image.shape[0];

print('shape of training image:', train_image.shape)
print('shape of training label:', train_label.shape)
print('shape of test image:', test_image.shape)
print('shape of test label:', test_label.shape)

# initial value of the coefficients
x = np.zeros( (n,1) )

"""
Computer the coefficients x in the logistic regressor!


"""

# print out the accuracy of your trained logistic regressor.
accuracy = np.sum( test_label == ( 1 / ( 1 + np.exp( np.dot( -np.transpose(x), test_image ) ) ) > 0.5) ) / test_label.size;
print('Training accuracy: {0:.1f}%'.format(100*accuracy))