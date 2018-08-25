# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:45:45 2018

@author: Artem Los
"""

import numpy as np
from helpers import *
from methods import OneLayerNetwork
import matplotlib.pyplot as plt

# Initialize the datasets
X_train, Y_train, _ = format_data(get_data_and_labels('Datasets/data_batch_1'))
X_val, Y_val, _ = format_data(get_data_and_labels('Datasets/data_batch_2'))
X_test, Y_test, _ = format_data(get_data_and_labels('Datasets/test_batch'))

np.random.seed(4711)

# Initializing the one layer network
clf = OneLayerNetwork(n_epochs=10, l=1, eta=0.01 )

res = clf.fit(X_train,Y_train, X_test, Y_test)
train_scores = np.vstack(res[0])
test_scores = np.vstack(res[1])

def plot_results(train_scores, test_scores):
    plt.figure(1)
    
    plt.title("Loss vs. epochs")
    plt.plot(train_scores, label='training set')
    plt.plot(test_scores, label='validation set')
    plt.legend()
    
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    
    plt.xticks([0,10,20,30,40,50])
    plt.savefig('Report/img/training.png')
    
print("Training set:")
print(clf.compute_accuracy(X_train,Y_train))
print("Test set: ")
print(clf.compute_accuracy(X_test,Y_test))


def checks(n=1, s=0):
    
    np.random.seed(4711)
    clf = OneLayerNetwork(n_epochs=1, l = 0, eta=0.01, n_batch=n-s)
    num = clf.compute_grads_num_slow(X_train[:,s:n].reshape(3072,n-s), Y_train[:,s:n].reshape(10,n-s))
    an = clf.compute_gradients(X_train[:,s:n].reshape(3072,n-s), Y_train[:,s:n].reshape(10,n-s))
     
     
    print(np.sum(np.abs(res[1][0] - res[0][0])))
    print(np.sum(np.abs(res[1][1] - res[0][1])))
    
    W = clf.W - num[0]
    b = clf.b - num[1].T
     
    return (an,num, W, b)


def grad_diff(gn, ga):
    
    num = np.linalg.norm(ga - gn)
    den = np.maximum(1e-6, np.linalg.norm(ga)+np.linalg.norm(gn))
    
    return num/den

plot_results(train_scores, test_scores)
show_weights(clf)



 