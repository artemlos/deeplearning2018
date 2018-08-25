# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:10:05 2018

@author: Artem Los
"""

import numpy as np
from helpers import *

class KLayerNetwork():

    """
    d - number of features (input vector)
    K - number of output labels
    l - regression term lambda
    n_batch - number of mini-batches
    n_epochs - number of epochs
    m - the format of the hidden layer, eg (50, 30) or (50, )
    rho - the momentum constant (if None, momentum learning will not be used).
    decay_rate = used to reduce learning rate over time.
    
    Remarks:
        X is a matrix where each column is a data point.
        y should always be in one-hot notation.
    """
    def __init__(self, d=1024*3, K=10, l=0.01, eta = 0.01, n_batch = 100, n_epochs = 2, m=(50, ), rho = None, decay_rate=1, include_train_cost=True, batch_norm=True):
        
        self.W = []
        
        prev = d
        
        for m_i in m:
            self.W.append(self.__init_vectors(m_i, prev))
            prev = m_i
            
        # the last output layer
        self.W.append(self.__init_vectors(K, prev))
        
        self.b = []

        for m_i in m:
            self.b.append(self.__init_vectors(m_i, 1, useOnlyZeros=True))
            
        # the last output layer    
        self.b.append(self.__init_vectors(K, 1, useOnlyZeros=True))
        
        self.mu = []
        self.v = []
        for m_i in m:
            self.mu.append(np.zeros((m_i,)))
            self.v.append(np.zeros((m_i,)))
        
        self.l = l
        
        self.K = K
        self.d = d
        
        self.eta = eta
        self.n_batch = n_batch
        self.n_epochs = n_epochs
        
        self.batch_norm = batch_norm
        
        self.m = m
        
        self.rho = rho
        
        self.vW = []
        
        prev = d
        
        for m_i in m:
            self.vW.append(self.__init_vectors(m_i, prev, useOnlyZeros=True))
            prev = m_i
            
        self.vW.append(self.__init_vectors(K, prev, useOnlyZeros=True))
        
        
        self.vb = []
        
        for m_i in m:
            self.vb.append(self.__init_vectors(1, m_i, useOnlyZeros=True))
            
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
        
        for i in range(self.n_epochs):
            for j in range(X.shape[1]//self.n_batch):
                
                j_start = j*self.n_batch;
                j_end = (j+1)*self.n_batch
                
                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]
                
                self.__mini_batch_GD(X_batch, Y_batch)
 
            if self.include_train_cost:
                train_cost = self.j_cost_func(X,Y, use_pre_computed_mu_v=False)
                test_cost = self.j_cost_func(X_test,Y_test, use_pre_computed_mu_v=False)
                train_costs.append(train_cost)
                test_costs.append(test_cost)
                print(train_cost)
                print(test_cost)

                print("--------------------------")

            
            self.eta = self.eta * self.decay_rate
        
        return (train_costs,test_costs);

    """
    W and b will be updated automatically.
    
    Remarks:
        Ideally, we want to perform as much computations in numpy as possible.
    """    
    def __mini_batch_GD(self, X_batch, Y_batch):
        
        k = len(self.m)
        
        dW, db = [], []
        
        if self.batch_norm:
            dW, db = self.compute_gradients_batch_norm(X_batch, Y_batch)
        else:
            dW, db = self.compute_gradients(X_batch, Y_batch); #self.compute_grads_num_slow(X_batch, Y_batch)
                
        if self.rho != None:
            for i in range(k+1):
                self.vW[i] = self.rho*self.vW[i] + self.eta*dW[i]
                self.vb[i] = self.rho*self.vb[i] + self.rho*db[i]            
        else:
            for i in range(k+1):
                self.vW[i] = self.eta * dW[i]
                self.vb[i] = self.eta * db[i]
            
        for i in range(k+1):
            self.W[i] = self.W[i] - self.vW[i]
            self.b[i] = self.b[i] - self.vb[i].T
    
    """
    Computes gradients without batch norm
    """
    def compute_gradients(self, X_batch, y):
        
        k = len(self.m)
        
        # set all entries to zero
        dLdW = []        
        prev = self.d
        for m_i in self.m:
            dLdW.append(self.__init_vectors(m_i, prev, useOnlyZeros=True))
            prev = m_i
        dLdW.append(self.__init_vectors(self.K, prev, useOnlyZeros=True))

        dLdb = []
        for m_i in self.m:
            dLdb.append(np.zeros((1, m_i)))        
        dLdb.append(np.zeros((1, self.K)))        
        
        # for each data point
        for i in range(X_batch.shape[1]):
            
            x_i = X_batch[:, i].reshape(-1, 1) 
            y_i = y[:, i].reshape(-1, 1)
            
            p, s, x = self.evaluate_classifier_with_activations(x_i)
                    
            g = -(y_i-p).T
            
            # m=(50,) => 1,0
            for j in reversed(range(k+1)):
                dLdb[j] += g
                #should it be x[i] or x[i-1]? note, 
                #we added one param extra in forward pass
                dLdW[j] += g.T.dot(x[j].T)
                g = g.dot(self.W[j])
                
                # only back-propogate to next level if we have not
                # reached the first layer.
                if j > 0:
                    self.ind(s[j-1])
                    g = g.dot(np.diag(s[j-1].reshape(self.m[j-1])))
            
        dJdW = []
        for i in range(k+1):
            dLdW[i] /= X_batch.shape[1]
            dLdb[i] /= X_batch.shape[1]
            
            dJdW.append(dLdW[i]+2*self.l*self.W[i])
        
        return (dJdW, dLdb)
    
    """
    Computes gradients with batch norm
    """
    def compute_gradients_batch_norm(self, X_batch, y):
        
        k = len(self.m)
        
        # set all entries to zero
        dJdW = []        
        prev = self.d
        for m_i in self.m:
            dJdW.append(self.__init_vectors(m_i, prev, useOnlyZeros=True))
            prev = m_i
        dJdW.append(self.__init_vectors(self.K, prev, useOnlyZeros=True))

        dJdb = []
        for m_i in self.m:
            dJdb.append(np.zeros((1, m_i)))        
        dJdb.append(np.zeros((1, self.K)))
        
        # forward pass
        p, inter = self.eval_classifier(X_batch)
        
        X = inter[2]
        S = inter[0]
        S_hat = inter[1]
        mu = inter[3]
        v = inter[4]
        
        # keeping an expontential moving average of the mean and variance
        for i in range(k):
            self.update_param('mu', i, mu[i])
            self.update_param('v', i, v[i])
        
        
        # backward pass
        # for each data point
        g = [] # stores a list of eg. (1,10).
        gTxT = []
        for i in range(X_batch.shape[1]):
            
            y_i = y[:, i].reshape(-1,1)
            p_i = p[:, i].reshape(-1,1)
            x_i = X[k][:,i].reshape(-1,1) 
            
            g.append(-(y_i-p_i).T)
            
            gTxT.append(g[-1].T.dot(x_i.T))
            
        # the sums for the last layer
        dJdb[k] = np.mean(g, axis=0)
        dJdW[k] = np.mean(gTxT, axis=0) + 2*self.l*self.W[k]
        
        # propogate to the previous layer
        for i in range(X_batch.shape[1]): # could possibly be put in prev loop
            g[i] = g[i].dot(self.W[k])
            self.ind(S_hat[k-1][:, i]) # or should this be final layer without softmax?, eg S[k]
            g[i] = g[i].dot(np.diag(S_hat[k-1][:, i]))
        
        for l in reversed(range(k)):
            self.batch_norm_back_pass(g, S[l], mu[l], v[l])
            
            dJdb[l] = np.mean(g, axis=0)
            
            gTxT = []
            
            for i in range(X_batch.shape[1]):
                
                x_i = X[l][:,i].reshape(-1,1) 
                
                gTxT.append(g[i].T.dot(x_i.T))
            
            dJdW[l] = np.mean(gTxT, axis=0) + 2*self.l*self.W[l]
            
            # propogate to the previous layer (if l > 1)
            if l > 0:
                for i in range(X_batch.shape[1]): # could possibly be put in prev loop
                    g[i] = g[i].dot(self.W[l])
                    self.ind(S_hat[l-1][:, i])
                    g[i] = g[i].dot(np.diag(S_hat[l-1][:, i]))
                    
        return (dJdW, dJdb)
    
    """
    Applied layerwise
    """
    def batch_norm_back_pass(self, g, s, mu, v):
        
        dJdvb = np.zeros(v.shape).reshape(1,-1)
        dJdmub = np.zeros(mu.shape).reshape(1,-1)
        
        for i in range(len(g)):
        
            dJdvb += -0.5*(g[i]/np.sqrt(v**3+1e-8)).dot(np.diag(s[:,i] - mu)) 
            dJdmub += g[i]/np.sqrt(v+1e-8)
            
        for i in range(len(g)):
            g[i] = g[i]/np.sqrt(v+1e-8) + 2/len(g) * dJdvb.dot(np.diag(s[:, i] - mu)) + dJdmub*(1/len(g))
        

    def ind(self, x):
        x[x>0] = 1
        x[x<=0] = 0
        
     
    """
    Uses exponential moving average to keep track of the mu and v during training.
    """
    def update_param(self, name, layer, new_value):
        
        alpha = 0.99
        
        #empty = np.zeros((self.m[layer],))
        
        #if (self.__dict__[name][layer] == empty).all():
        #    self.__dict__[name][layer] = new_value
        #else:
        self.__dict__[name][layer] = alpha*self.__dict__[name][layer] + (1-alpha)*new_value
        

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
    Should take in the whole mini-batch (for batch norm)
    """
    def eval_classifier(self, X_batch, use_pre_computed_mu_v=False):
                
        k = len(self.m)
        
        S = []
        S_hat = []
        X = [X_batch]
        mu = []
        v = []

        
        if(use_pre_computed_mu_v):
            # mean and var are specified
            mu = self.mu
            v = self.v
        
        # for each i = 1,.., k-1
        for i in range(k):
    
            s_temp = np.zeros((self.W[i].shape[0], X_batch.shape[1]))
            
            # for all examples in D
            for j in range(X_batch.shape[1]):
                s_temp[:, j] = (self.W[i].dot(X[i][:,j].reshape(-1,1)) + self.b[i]).flatten()
            
            S.append(s_temp)
            
            if not(use_pre_computed_mu_v):
                mu.append(np.mean(s_temp, axis=1))
                v.append(np.var(s_temp, axis=1)) #* (n-1)/n)     
            
            s_hat_temp = np.zeros((self.W[i].shape[0], X_batch.shape[1]))
            x_temp = np.zeros((self.W[i].shape[0], X_batch.shape[1]))
            
            for j in range(X_batch.shape[1]):
                s_hat_temp[:,j] = (self.batch_normalize(s_temp[:,j], mu[i], v[i])).flatten()
                x_temp[:,j] = np.maximum(s_hat_temp[:, j],0)
            
            S_hat.append(s_hat_temp)
            X.append(x_temp)
            
        final_layer = np.zeros((self.W[k].shape[0], X_batch.shape[1]))    
     
        for j in range(X_batch.shape[1]):
            final_layer[:,j] = (self.__softmax2(self.W[k].dot(X[-1][:, j].reshape(-1,1)) + self.b[k])).flatten()
            
        return final_layer, (S,S_hat, X, mu, v)        
            
    
    """
    The vector of proabilities, p, for each output label.
    """
    def evaluate_classifier_with_activations(self, X, mu=None, v=None):  
        # note, we added an extra param to x, which might affect indexsation
        s = []
        
        x = [X]
        
        k = len(self.m)
        
        for i in range(k):
            s.append(self.W[i].dot(x[i]) + self.b[i])
            x.append(np.maximum(s[-1],0))
            
        return self.__softmax2(self.W[k].dot(x[-1]) + self.b[k]), s, x
        
    
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
    
    
    def batch_normalize(self, s, mu, v):
        V = np.diag(v+1e-8) # maybe we can skip diag and use the same trick as in BN in FP.

        return (s-mu)/np.sqrt(v+1e-8) #(V**(-0.5)).dot(s-mu) #
        #return np.power(np.diag(v+eps), -0.5).dot(s-mu)
    
    """
    Compute the cost function (includes a regression term).
    """
    def j_cost_func(self, X, y, use_pre_computed_mu_v=True):
        
        k = len(self.m)
        
        l_cross = 0
        
        if self.batch_norm:
            # should mean and var be included?
            
            p, inter = self.eval_classifier(X, use_pre_computed_mu_v)

            for i in range(X.shape[1]):
                l_cross -= np.log(np.dot(y[:,i].reshape(1, -1),p[:,i].reshape(-1,1)))  
            
        else:
            for i in range(X.shape[1]):
                l_cross -= np.log(np.dot(y[:,i].reshape(-1, 1).T,self.evaluate_classifier(X[:,i].reshape(-1,1))))       

        reg_term = 0

        for i in range(k+1):
            reg_term += self.l*np.sum(np.square(self.W[i]))
        
        # reg_term = self.l*(self.l*np.sum(np.square(self.W[0])) + self.l*np.sum(np.square(self.W[1])))
        
        return (1/X.shape[1] * l_cross + reg_term)[0][0]
    
    """
    Computes the accuracy of the model.
    """
    def compute_accuracy(self, X, y, use_pre_computed_mu_v=True):
        
        # TODO: instead of argmax, maybe better to just use 
        # (predicted == actual).sum()
        predicted = []
        
        if self.batch_norm:
            predicted = np.argmax(self.eval_classifier(X, use_pre_computed_mu_v)[0], axis=0)
        else:        
            predicted = np.argmax(self.evaluate_classifier(X), axis=0)
            
        actual = np.argmax(y, axis=0)
        
        return np.sum(actual == predicted) / len(predicted)
        
