# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:59:07 2018

@author: Artem Los
"""

from RNN import RNN
from grad import Grad

import numpy as np

import copy

rnn = RNN()

rnn.load_data()

#a, _, _ = rnn.get_next(rnn.get_one_hot(5), rnn.get_h_0())

a,b, _,_,_,_ = rnn.get_seq(rnn.get_one_hot(5), rnn.get_h_0(),25)

print(list(map(lambda x: rnn.ind_to_char[x], b)))

print(rnn.cost_func())

h_0 = rnn.get_h_0()

_, _, P, H, O, A = rnn.get_seq(rnn.X_chars[:,0], h_0, rnn.seq_length)

dLdo = -(rnn.Y_chars-P).T
dLdh_t= dLdo[rnn.seq_length-1].dot(rnn.V)

#grads = rnn.fit()

analytic = None

def check_gradients():
    
    global rnn, analytic
    
    rnn = RNN(m=5)
    rnn.load_data()
    
    analytic = rnn.get_grad(rnn.get_h_0())

    # you need to run this code for each param separatly.
    for grad in ['W']:
        print("started on " + grad)
        rnn.__dict__[grad] = compute_grad(grad, rnn)        
    
    
def compute_grad(f, rnn):
    
    h = 1e-4
    
    n = np.size(rnn.__dict__[f])
    grad = np.zeros(rnn.__dict__[f].shape)
    
    hprev = np.zeros((rnn.W.shape[0], 1))
    
    for i in range(n):

        rnn_try = copy.deepcopy(rnn)
        
        rnn_try.__dict__[f].itemset(i, rnn.__dict__[f].item(i) - h);
        
        # not sure if this cost func is correct, as it tries to generate new seq
        # whereas this should be more like the forward pass
        
        l1 = rnn_try.cost_func2(hprev) # TODO, fix so that it does not generate new X
        
        rnn_try.__dict__[f].itemset(i, rnn.__dict__[f].item(i) + h);
        
        l2 = rnn_try.cost_func2(hprev)
        
        grad.itemset(i, (l2-l1)/(2*h));
        
    return grad

def grad_diff(gn, ga):
    
    num = np.linalg.norm(ga - gn)
    den = np.maximum(1e-6, np.linalg.norm(ga)+np.linalg.norm(gn))
    
    return num/den
        
        
check_gradients()  
print(grad_diff(analytic[0].dLdb, rnn.b.T))
print(grad_diff(analytic[0].dLdc, rnn.c))
print(grad_diff(analytic[0].dLdU, rnn.U))
print(grad_diff(analytic[0].dLdW, rnn.W))
print(grad_diff(analytic[0].dLdV, rnn.V))

# transpose all the weights, maybe not necessary.
    
    
    
    
    
    
    
    