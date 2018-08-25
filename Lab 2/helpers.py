# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:22:25 2018

@author: Artem Los
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
    

"""
Pick a permissible random learning rate.
"""
def rnd_eta_lambda(min_val=-4, max_val=-1):
    #e = min_val + (max_val-min_val)*np.random.rand()
    return np.power(10, np.random.uniform(min_val, max_val))


"""
Performs required pre-formatting of the data.
"""
def format_data(data):
    return (data[0].T/255, np_utils.to_categorical(data[1]).T ,data[1]);

"""
Returns the Data & Label pair for a given CIFAR file.
"""
def get_data_and_labels(file):
    dataset = unpickle(file)    
    return (dataset[b'data'], dataset[b'labels'])

"""
Used to extract the files from the CIFAR dataset.
From: https://www.cs.toronto.edu/~kriz/cifar.html
"""
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
Show 25 random images from CIFAR dataset.
From: https://stackoverflow.com/a/40144107
"""
def show_random_images(images, dim=5):

    X = images.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    
    fig, axes1 = plt.subplots(5,5,figsize=(dim,dim))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X[i:i+1][0])
    
    plt.show()
    
    
"""
Show the weights of a classifier.
"""
def show_weights(clf):
    fig, axes1 = plt.subplots(1,10, figsize=(8,8))

    for i in range(10):
        im = clf.W[i].reshape(32,32,3)
        #im = (1/(2*2.25)) * im + 0.5
        im = (im - np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
        
        
        axes1[i].set_axis_off()
        axes1[i].imshow(im)
 
    fig.savefig('Report/img/weights.png')  