# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:10:05 2018

@author: Artem Los
"""

import numpy as np
from helpers import *

class OneLayerNetwork():

    """
    d - number of features (input vector)
    K - number of output labels
    l - regression term lambda
    
    Remarks:
        X is a matrix where each column is a data point.
        y should always be in one-hot notation.
    """
    def __init__(self, d=1024*3, K=10, l=0.01, eta = 0.01, n_batch = 100, n_epochs = 2):
        
        self.W = self.__init_vectors(K, d)
        self.b = self.__init_vectors(K, 1)
        self.l = l
        
        self.K = K
        self.d = d
        
        self.eta = eta
        self.n_batch = n_batch
        self.n_epochs = n_epochs
        
        return;
        
    """
    Train the network on the training set.
    """    
    def fit(self, X, Y, X_test, Y_test):
        
        train_costs = [];
        test_costs = [];
        
        for i in range(self.n_epochs):
            for j in range(X.shape[1]//self.n_batch):
                
                j_start = j*self.n_batch;
                j_end = (j+1)*self.n_batch
                
                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]
                
                self.__mini_batch_GD(X_batch, Y_batch)
 
            train_costs.append(self.j_cost_func(X,Y))
            test_costs.append(self.j_cost_func(X_test,Y_test))
            
        return (train_costs,test_costs);

    """
    W and b will be updated automatically.
    """    
    def __mini_batch_GD(self, X_batch, Y_batch):
        
        dW, db = self.compute_gradients(X_batch, Y_batch); #self.compute_grads_num_slow(X_batch, Y_batch)
        
        self.W = self.W - self.eta * dW
        self.b = self.b - self.eta * db.T# db.reshape(self.K, 1) # or transpose?
        
    
    # internal function
    def compute_gradients(self, X, y):
        
        # set all entries to zero
        dLdW = np.zeros((self.K, self.d))
        dLdb = np.zeros((1, self.K))
                
        # for each data point
        for i in range(X.shape[1]):
            
            x_i = X[:, i].reshape(-1, 1) 
            y_i = y[:, i].reshape(-1, 1)
            
            p = self.evaluate_classifier(x_i)
                    
            g = -(y_i-p).T
            
            dLdb += g
            dLdW += g.T.dot(x_i.T)
            
        dLdW /= X.shape[1]
        dLdb /= X.shape[1]
                
        return (dLdW+2*self.l*self.W, dLdb)
    
    
    
    def compute_grads_num_slow(self, X, y, h = 0.001):
        
        dLdW = np.zeros((self.K, self.d))
        dLdb = np.zeros((1, self.K))
        

        for i in range(self.b.shape[1]):
            b_old = np.copy(self.b)
            
            self.b[i] = self.b[i] - h;
            c1 = self.j_cost_func(X, y)
            self.b = np.copy(b_old)
            self.b[i] = self.b[i] + h;
            c2 = self.j_cost_func(X, y)
            
            self.b = np.copy(b_old)
            dLdb[i] = (c2-c1) / (2*h)
            
        for i in range(np.size(dLdW)):
            W_old = np.copy(self.W)
            
            self.W.itemset(i, self.W.item(i) - h);
            c1 = self.j_cost_func(X, y)
            self.W = np.copy(W_old)
            self.W.itemset(i, self.W.item(i) + h);
            c2 = self.j_cost_func(X, y)
            
            self.W = np.copy(W_old)
            dLdW.itemset(i, (c2-c1) / (2*h))
        
        return (dLdW, dLdb);
            
    
    """
    The vector of proabilities, p, for each output label.
    """
    def evaluate_classifier(self, X):  
        
        return self.__softmax2(self.W.dot(X) + self.b)
    
    def __softmax(self, s):

        return np.exp(s)/(np.sum(np.exp(s), axis=0))
    
    
    def __softmax2(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def __init_vectors(self, row, col):
        
        return np.random.normal(loc=0.0, scale=0.01, size=(row,col));
    
    """
    Compute the cost function (includes a regression term).
    """
    def j_cost_func(self, X, y):
        
        #l_cross= np.sum(-np.log(y.T.dot(self.evaluate_classifier(X))))
        
        l_cross = 0
        
        for i in range(X.shape[1]):
            l_cross -= np.log(np.dot(y[:,i].reshape(-1, 1).T,self.evaluate_classifier(X[:,i].reshape(-1,1))))       
                
        return 1/X.shape[1] * l_cross + self.l*np.sum(np.square(self.W))
    
    """
    Computes the accuracy of the model.
    """
    def compute_accuracy(self, X, y):
        
        # TODO: instead of argmax, maybe better to just use 
        # (predicted == actual).sum()
        
        predicted = np.argmax(self.evaluate_classifier(X), axis=0)
        actual = np.argmax(y, axis=0)
        
        return np.sum(actual == predicted) / len(predicted)
        
