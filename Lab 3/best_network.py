# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 21:46:21 2018

@author: Artem Los
"""


import numpy as np
import numpy.matlib
from helpers import *
from methods import TwoLayerNetwork
import matplotlib.pyplot as plt
import pickle

# Initialize the datasets
X_train1, Y_train1, _ = format_data(get_data_and_labels('Datasets/data_batch_1'))
X_train2, Y_train2, _ = format_data(get_data_and_labels('Datasets/data_batch_2'))
X_train3, Y_train3, _ = format_data(get_data_and_labels('Datasets/data_batch_3'))
X_train4, Y_train4, _ = format_data(get_data_and_labels('Datasets/data_batch_4'))
X_train5, Y_train5, _ = format_data(get_data_and_labels('Datasets/data_batch_5'))

X_test, Y_test, _ = format_data(get_data_and_labels('Datasets/test_batch'))

X_train = np.hstack((X_train1, X_train2, X_train3, X_train4, X_train5))
Y_train = np.hstack((Y_train1, Y_train2, Y_train3, Y_train4, Y_train5))

X_val, Y_val = X_train[:, 49000:50000], Y_train[:, 49000:50000]
X_train, Y_train = X_train[:, 0:49000], Y_train[:, 0:49000]


mean_X = np.mean(X_train, axis=1).reshape(-1,1)
X_train = X_train - np.matlib.repmat(mean_X, 1, X_train.shape[1])
X_val = X_val - np.matlib.repmat(mean_X, 1, X_val.shape[1])
X_test = X_test - np.matlib.repmat(mean_X, 1, X_test.shape[1])


clf = TwoLayerNetwork(n_epochs=30, l=1.49535682e-04, eta=1.60596650e-02, 
                      n_batch=100, decay_rate=0.95, rho=0.9)

train, test = clf.fit(X_train, Y_train, X_val, Y_val)


def plot_results(train_scores, test_scores):
    plt.figure(1)
    
    plt.title("Loss vs. epochs")
    plt.plot(train_scores, label='training set')
    plt.plot(test_scores, label='validation set')
    plt.legend()
    
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    
    #plt.xticks([0,10,20,30,40,50])
    plt.savefig('Report/img/training.png')