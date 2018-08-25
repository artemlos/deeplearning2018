# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:10:05 2018

@author: Artem Los
"""

import numpy as np
from helpers import *

class TwoLayerNetwork():

    """
    d - number of features (input vector)
    K - number of output labels
    l - regression term lambda
    n_batch - number of mini-batches
    n_epochs - number of epochs
    m - the size of the hidden layer
    rho - the momentum constant (if None, momentum learning will not be used).
    decay_rate = used to reduce learning rate over time.
    
    Remarks:
        X is a matrix where each column is a data point.
        y should always be in one-hot notation.
    """
    def __init__(self, d=1024*3, K=10, l=0.01, eta = 0.01, n_batch = 100, n_epochs = 2, m=50, rho = None, decay_rate=1, include_train_cost=True):
        
        self.W = []
        self.W.append(self.__init_vectors(m, d))
        self.W.append(self.__init_vectors(K, m))
        
        self.b = []
        self.b.append(self.__init_vectors(m, 1, useOnlyZeros=True))
        self.b.append(self.__init_vectors(K, 1, useOnlyZeros=True))
        
        self.l = l
        
        self.K = K
        self.d = d
        
        self.eta = eta
        self.n_batch = n_batch
        self.n_epochs = n_epochs
        
        self.m = m
        
        self.rho = rho
        
        self.vW = []
        self.vW.append(self.__init_vectors(m, d, useOnlyZeros=True))
        self.vW.append(self.__init_vectors(K, m, useOnlyZeros=True))
        
        self.vb = []
        self.vb.append(self.__init_vectors(1, m, useOnlyZeros=True))
        self.vb.append(self.__init_vectors(1, K, useOnlyZeros=True))
        
        self.decay_rate = decay_rate
        
        self.include_train_cost = include_train_cost
        
        return;
        
    """
    Train the network on the training set.
    """    
    def fit(self, X, Y, X_test=None, Y_test=None):
        
        train_costs = [];
        test_costs = [];
        
        org_train_cost = self.j_cost_func(X, Y)
        
        for i in range(self.n_epochs):
            for j in range(X.shape[1]//self.n_batch):
                
                j_start = j*self.n_batch;
                j_end = (j+1)*self.n_batch
                
                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]
                
                self.__mini_batch_GD(X_batch, Y_batch)
 
            if self.include_train_cost:
                train_costs.append(self.j_cost_func(X,Y))
                test_costs.append(self.j_cost_func(X_test,Y_test))
                
            if self.j_cost_func(X, Y) > 3 * org_train_cost:
                return None;
            
            self.eta = self.eta * self.decay_rate
            
        return (train_costs,test_costs);

    """
    W and b will be updated automatically.
    
    Remarks:
        Ideally, we want to perform as much computations in numpy as possible.
    """    
    def __mini_batch_GD(self, X_batch, Y_batch):
        
        dW, db = self.compute_gradients(X_batch, Y_batch); #self.compute_grads_num_slow(X_batch, Y_batch)
                
        if self.rho != None:
            self.vW[0] = self.rho*self.vW[0] + self.eta*dW[0]
            self.vb[0] = self.rho*self.vb[0] + self.rho*db[0]
            self.vW[1] = self.rho*self.vW[1] + self.eta*dW[1]
            self.vb[1] = self.rho*self.vb[1] + self.rho*db[1]
            
        else:
            self.vW[0] = self.eta * dW[0]
            self.vW[1] = self.eta * dW[1]    
            self.vb[0] = self.eta * db[0]
            self.vb[1] = self.eta * db[1]
            
        self.W[0] = self.W[0] - self.vW[0]
        self.b[0] = self.b[0] - self.vb[0].T
        self.W[1] = self.W[1] - self.vW[1]
        self.b[1] = self.b[1] - self.vb[1].T
        
    
    # internal function
    def compute_gradients(self, X, y):
        
        # set all entries to zero
        dLdW = []
        dLdW.append(np.zeros((self.m, self.d)))
        dLdW.append(np.zeros((self.K, self.m)))

        dLdb = []        
        dLdb.append(np.zeros((1, self.m)))
        dLdb.append(np.zeros((1, self.K)))
                
        # for each data point
        for i in range(X.shape[1]):
            
            x_i = X[:, i].reshape(-1, 1) 
            y_i = y[:, i].reshape(-1, 1)
            
            p, s1, h = self.evaluate_classifier_with_activations(x_i)
                    
            g = -(y_i-p).T
            
            dLdb[1] += g
            
            dLdW[1] += g.T.dot(h.T)
            
            g = g.dot(self.W[1])
            
            self.ind(s1)
            
            g = g.dot(np.diag(s1.reshape(self.m)))
            
            dLdb[0] += g
            dLdW[0] += g.T.dot(x_i.T)
            
            
            
        dLdW[0] /= X.shape[1]
        dLdb[0] /= X.shape[1]
        
        dLdW[1] /= X.shape[1]
        dLdb[1] /= X.shape[1]
                
        dJdW = [dLdW[0]+2*self.l*self.W[0], dLdW[1]+2*self.l*self.W[1]]
        
        return (dJdW, dLdb)
    

    def ind(self, x):
        x[x>0] = 1
        x[x<=0] = 0
        

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
            
    
    def evaluate_classifier(self, X):
        res, _, _ = self.evaluate_classifier_with_activations(X)
        return res
    
    """
    The vector of proabilities, p, for each output label.
    """
    def evaluate_classifier_with_activations(self, X):  
        
        s1 = self.W[0].dot(X) + self.b[0]
        h = np.maximum(s1,0) #relu
        return self.__softmax2(self.W[1].dot(h) + self.b[1]), s1, h
        
    
    def __softmax(self, s):

        return np.exp(s)/(np.sum(np.exp(s), axis=0))
    
    
    def __softmax2(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def __init_vectors(self, row, col, useOnlyZeros = False):
        
        if not(useOnlyZeros):
            return np.random.normal(loc=0.0, scale=0.001, size=(row,col));
        return np.zeros((row,col))
    
    """
    Compute the cost function (includes a regression term).
    """
    def j_cost_func(self, X, y):
        
        l_cross = 0
        
        for i in range(X.shape[1]):
            l_cross -= np.log(np.dot(y[:,i].reshape(-1, 1).T,self.evaluate_classifier(X[:,i].reshape(-1,1))))       
                
        reg_term = self.l*(self.l*np.sum(np.square(self.W[0])) + self.l*np.sum(np.square(self.W[1])))
        
        return 1/X.shape[1] * l_cross + reg_term
    
    """
    Computes the accuracy of the model.
    """
    def compute_accuracy(self, X, y):
        
        # TODO: instead of argmax, maybe better to just use 
        # (predicted == actual).sum()
        
        predicted = np.argmax(self.evaluate_classifier(X), axis=0)
        actual = np.argmax(y, axis=0)
        
        return np.sum(actual == predicted) / len(predicted)
        
