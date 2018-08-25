# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 14:45:10 2018

@author: Artem Los
"""
import numpy as np
from keras.utils import np_utils

from grad import Grad

class RNN:
    
    """
    K - the size of the input (eg. size of the alphabet)
    m - the size of the hidden layer
    """    
    def __init__(self, eta = 0.01, K=83, seq_length = 25, m = 100, sig = 0.01, epochs=100):
        
        self.b = np.zeros((m, 1))
        self.c = np.zeros((K, 1))
        
        self.U = self.__init_vectors(m, K) * sig
        self.W = self.__init_vectors(m, m) * sig
        self.V = self.__init_vectors(K, m) * sig
        
        self.dataset = None
        self.char_to_ind = None
        self.ind_to_char = None

        self.K = K
        self.m = m
        
        self.eta=eta
        self.seq_length = seq_length
        self.sig = sig
        
        self.X_chars = None
        self.Y_chars = None
        
        self.epochs = epochs
        
        
    def fit(self):
        
        # better with while loop
        
        hprev = self.get_h_0()
        mb = 0
        mc = 0
        mW = 0
        mU = 0
        mV = 0
        
        self.X_chars = self.dataset[:, 0:self.seq_length]
        self.Y_chars = self.dataset[:, 1:self.seq_length+1]
        
        smooth_loss = self.cost_func()
        smooth_losses = []
        # self.epochs*len(self.dataset)
        # TODO: we need to take into account the condition in lab description.
        # self.dataset.shape[1]-1
        
        
        for i in range(2):
            
            e = 0
            hprev=self.get_h_0()
            
            while e < self.dataset.shape[1]-self.seq_length-1:
    
                self.X_chars = self.dataset[:, e:self.seq_length+e]
                self.Y_chars = self.dataset[:, (e+1):self.seq_length+e+1]
                
                grads, H, fw_loss = self.get_grad(hprev)
                
                # AdaGrad
                
                mb = mb + grads.dLdb**2
                mc = mc + grads.dLdc**2
                mW = mW + grads.dLdW**2
                mU = mU + grads.dLdU**2
                mV = mV + grads.dLdV**2
                
                eps = 1e-1 #1e-4 #1e-1 for faster learning # from lecture 5
                
                self.b = self.b - np.multiply(self.eta/np.sqrt(mb + eps), grads.dLdb).T
                self.c = self.c - np.multiply(self.eta/np.sqrt(mc + eps), grads.dLdc)
                self.W = self.W - np.multiply(self.eta/np.sqrt(mW + eps), grads.dLdW)
                self.U = self.U - np.multiply(self.eta/np.sqrt(mU + eps), grads.dLdU)
                self.V = self.V - np.multiply(self.eta/np.sqrt(mV + eps), grads.dLdV)
            
#                self.b = self.b - self.eta * grads.dLdb.T
#                self.c = self.c - self.eta * grads.dLdc
#                self.W = self.W - self.eta * grads.dLdW
#                self.U = self.U - self.eta * grads.dLdU
#                self.V = self.V - self.eta * grads.dLdV

                
                
                hprev = H[:,-1].reshape(-1, 1)
                
                smooth_loss = .999*smooth_loss + .001 * fw_loss
                    
                if e % 100 == 0:
                    smooth_losses.append(smooth_loss)
                
                if e % 10000 == 0:
                    print("Iteration: " + str(e), file=open("output3.txt", "a", encoding="utf-8"))
                    print("Smooth loss: " + str(smooth_loss), file=open("output3.txt", "a", encoding="utf-8"))
                    print(self.get_sample(self.X_chars[:, 0].reshape(-1,1), hprev), file=open("output3.txt", "a", encoding="utf-8"))
                    
                e += 1   
                
            if e % 10000 == 0:
                print("Iteration: " + str(e), file=open("output3.txt", "a", encoding="utf-8"))
                print("Smooth loss: " + str(smooth_loss), file=open("output3.txt", "a", encoding="utf-8"))
                print(self.get_sample(self.X_chars[:, 0].reshape(-1,1), hprev), file=open("output3.txt", "a", encoding="utf-8"))
            
        
        #print(self.cost_func())
        #print(self.get_sample(self.X_chars[:, 0].reshape(-1,1), hprev))
        
        return smooth_losses    
    
    def get_sample(self, x, h_0):
        a,b, _,_,_,_ = self.get_seq(x, h_0, 200)
        return "".join(list(map(lambda x: self.ind_to_char[x], b)))
        
        
    def get_seq(self, x_1, h_0, n):
        
        x_1 = x_1.reshape(-1, 1)
        h_0 = h_0.reshape(-1, 1)
        
        result = np.empty((self.K, 0))
        result2 = np.zeros(1)
        
        P = np.zeros((self.K, 0))
        H = np.zeros((self.m, 0))
        O = np.zeros((self.K, 0))
        A = np.zeros((self.m, 0))
        
        h = np.copy(h_0)
        x = np.copy(x_1)
        
        for i in range(n):
            x, h, p_t, o_t, a_t = self.get_next(x, h)
            result2 = np.hstack((result2, x))
            x = self.get_one_hot(x)
            result = np.hstack((result, x))
            
            P = np.hstack((P, p_t))
            H = np.hstack((H, h))
            O = np.hstack((O, o_t))
            A = np.hstack((A, a_t))
            
        return result, result2, P, H, O, A
    
    # x0?
    def get_next(self, x_t, h_tm):
        
        # TODO!!! self.U.dot(x_t).reshape(-1,1). Seems correct.
        a_t = self.W.dot(h_tm) + self.U.dot(x_t) + self.b
        h_t = np.tanh(a_t)
        o_t = self.V.dot(h_t) + self.c
        p_t =  self.__softmax2(o_t)
        
        cp = np.cumsum(p_t)
        a = np.random.rand()
        ixs = np.where(cp - a > 0)
        ii = ixs[0][0]
        return (ii, h_t, p_t, o_t, a_t)
    
    
    def load_data(self, file = 'goblet_book.txt'):
        book_data = []
        with open(file,encoding="utf-8") as f:
            book_data = list(f.read())
            
        book_chars = list(set(book_data))
        
        self.K = len(book_chars)
        
        self.__init__(eta=self.eta, K=self.K, seq_length = self.seq_length,
                      m=self.m, sig=self.sig)

        self.char_to_ind = dict(zip(book_chars, [x for x in range(len(book_chars))]))
        self.ind_to_char = { self.char_to_ind[x] : x for x in book_chars}
        
        self.dataset = np_utils.to_categorical(list(map(lambda x: self.char_to_ind[x], book_data)), self.K).T
        
        self.X_chars = self.dataset[:, 0:self.seq_length]
        self.Y_chars = self.dataset[:, 1:self.seq_length+1]
        
    def cost_func(self, *args):
        
        l_cross = 0
 
        p_t = []
        
        if len(args) == 0:
            _ , _, p_t, _, _ , _ = self.get_seq(self.X_chars[:, 0].reshape(-1,1), self.get_h_0(), self.seq_length)
        elif len(args) == 1:
            p_t = args[0]
        
        for i in range(self.seq_length):
            l_cross -= np.log(np.dot(self.Y_chars[:,i].reshape(-1, 1).T,p_t[:,i].reshape(-1,1)))
        
        return l_cross
    
    def cost_func2(self, h_0):
        
        P = np.zeros((self.K, 0))
        H = np.zeros((self.m, 0))
        O = np.zeros((self.K, 0))
        A = np.zeros((self.m, 0))
        
        h_t = np.copy(h_0)
        
        l_cross = 0

        # forward pass        
        for t in range(self.X_chars.shape[1]):            
            a_t = self.W.dot(h_t) + self.U.dot(self.X_chars[:, t]).reshape(-1,1) + self.b
            h_t = np.tanh(a_t)
            o_t = self.V.dot(h_t) + self.c
            p_t =  self.__softmax2(o_t)
            
            P = np.hstack((P, p_t))
            H = np.hstack((H, h_t))
            O = np.hstack((O, o_t))
            A = np.hstack((A, a_t))
            
            l_cross -= np.log(np.dot(self.Y_chars[:,t].reshape(-1, 1).T, p_t.reshape(-1,1)))
            
        return l_cross
    
    """
    Using currently loaded data. Rename to get_grad
    """
    def get_grad(self, h_0):
        
        # Some tips reading the code:
        # 1dim vector -> reshape(-1,1) => create a column vector
        # 1dim vector -> reshape(1,-1) => transpose the vector above
        
        # When implementing the gradients, the equations from pp. 370-380 in 
        # the Deep Learning book were used https://www.deeplearningbook.org/contents/rnn.html

        P = np.zeros((self.K, 0))
        H = np.zeros((self.m, 0))
        O = np.zeros((self.K, 0))
        A = np.zeros((self.m, 0))
        
        h_t = np.copy(h_0)
        
        l_cross = 0

        # forward pass        
        for t in range(self.X_chars.shape[1]):            
            a_t = self.W.dot(h_t) + self.U.dot(self.X_chars[:, t]).reshape(-1,1) + self.b
            h_t = np.tanh(a_t)
            o_t = self.V.dot(h_t) + self.c
            p_t =  self.__softmax2(o_t)
            
            P = np.hstack((P, p_t))
            H = np.hstack((H, h_t))
            O = np.hstack((O, o_t))
            A = np.hstack((A, a_t))
            
            l_cross -= np.log(np.dot(self.Y_chars[:,t].reshape(-1, 1).T, p_t.reshape(-1,1)))
        
                
        dLdo = -(self.Y_chars-P).T
        
        dLdh = np.zeros((self.seq_length, self.m))

        tau = self.seq_length-1

        # case for t = tau = seq_length -1
        dLdh[tau] = dLdo[tau].dot(self.V) # see comment below but for dLdo
        #dLda[self.seq_length-1] = dLdh[tau].dot(RNN.diff_tanh(A[:, tau])) # TODO: do we need reshap for dLdh?
        
        # to avoid creating a new loop
    
        dLdV = dLdo[tau].reshape(-1,1).dot(H[:, tau].reshape(1, -1))
        dLdW = dLdh[tau].reshape(1,-1).dot(np.diag(1-(H[:, tau]**2))).T.dot(H[:, tau-1].reshape(1,-1))
        dLdU = dLdh[tau].reshape(1,-1).dot(np.diag(1-(H[:, tau]**2))).T.dot(self.X_chars[:,tau].reshape(1,-1))   #dLda[self.seq_length-1].reshape(-1,1).dot(self.X_chars[:, self.seq_length-1].reshape(-1,1).T)

        
        for i in range(tau-1, -1, -1):
            
            dLdh[i] = dLdo[i].dot(self.V) + dLdh[i+1].reshape(-1,1).T.dot(np.diag(1-(H[:, i+1]**2))).dot(self.W) #dLda[i+1].dot(self.W)
            #dLda[i] = dLdh[i].dot(RNN.diff_tanh(A[:,self.seq_length-1]))
            
            # to avoid creating a new loop
            dLdV += dLdo[i].reshape(-1,1).dot(H[:, i].reshape(1,-1))
            dLdW += dLdh[i].reshape(1,-1).dot(np.diag(1-(H[:, i]**2))).T.dot(H[:, i-1].reshape(1,-1)) #dLda[i].reshape(-1,1).dot(H[:, i-1].reshape(-1,1).T)
            dLdU += dLdh[i].reshape(1,-1).dot(np.diag(1-(H[:, i]**2))).T.dot(self.X_chars[:,i].reshape(1,-1)) #dLda[i].reshape(-1,1).dot(self.X_chars[:, i-1].reshape(-1,1).T)
            
    
        # issues with overflow when float32. try with decimal if error persists.
        dLdc = np.sum(dLdo, axis=0, dtype=np.float64).reshape(-1, 1)
        dLdb = np.zeros((1, self.m))
        
        for i in range(self.seq_length):
            dLdb += dLdh[i].reshape(1,-1).dot(np.diag(1-H.T[i]**2)) # need to reshape H, or it won't work?

        grads =  Grad(dLdh, dLdo, dLdV, dLdW, dLdU, dLdb, dLdc)
        
        for grad in grads.__dict__.values():
            np.clip(grad, -5, 5, out=grad)
            
        return grads, H, l_cross
        
    def diff_a(self, t):
        
        if t == -1:
            return np.zeros((100, 1))
        
        calc = self.diff_a(t-1)
        
        tanh = RNN.diff_tanh(self.__A[:, t-1]).flatten()
        
        # not sure if this can be done (i.e. element wise product between calc and tanh)
        result = self.W.dot(np.multiply(calc.flatten(),tanh).reshape(self.m, 1)) + 1
        
        self.dadb = np.hstack((self.dadb, result))
        return result
    
    
    def diff_a_2(self, t):
        if t == -1:
            return np.zeros((100, 1))
        
        calc = self.diff_a(t-1)
        tanh = RNN.diff_tanh(self.__A[:, t-1]).flatten()
        
        result = self.W.dot(np.multiply(calc.flatten(),tanh).reshape(self.m, 1))
        
        self.dadc = np.hstack((self.dadc, result))
        return result
        
    def diff_o(self, t):
        
        return
    
    def diff_tanh(x):
        x = x.reshape(-1, 1)
        return 1 - np.square(np.tanh(x)) # TODO: use identity instead of 1 (already so in np by default)
    
    def get_one_hot(self, x):
        return np_utils.to_categorical(x, self.K).reshape(self.K, 1)
    
    def get_h_0(self):
        return np.zeros((self.m, 1))
    
    
    def __softmax2(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def __init_vectors(self, row, col):
        return np.random.randn(row, col) #np.random.normal(loc=0.0, scale=1, size=(row,col));