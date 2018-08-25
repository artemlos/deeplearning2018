# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:45:45 2018

@author: Artem Los
"""

import numpy as np
import numpy.matlib
from helpers import *
from methods import KLayerNetwork
import matplotlib.pyplot as plt
import pickle



# Initialize the datasets
X_train, Y_train, _ = format_data(get_data_and_labels('Datasets/data_batch_1'))
X_val, Y_val, _ = format_data(get_data_and_labels('Datasets/data_batch_2'))
X_test, Y_test, _ = format_data(get_data_and_labels('Datasets/test_batch'))

#np.random.seed(4711)

# Initializing the one layer network
#clf = OneLayerNetwork(n_epochs=10, l=1, eta=0.01 )

#knn.eval_classifier(X_train[:,0:100], use_pre_computed_mu_v=True)


mean_X = np.mean(X_train, axis=1).reshape(-1,1)
X_train = X_train - np.matlib.repmat(mean_X, 1, X_train.shape[1])
X_val = X_val - np.matlib.repmat(mean_X, 1, X_val.shape[1])
X_test = X_test - np.matlib.repmat(mean_X, 1, X_test.shape[1])


#knn.compute_accuracy(X_train,Y_train)


#knn = KLayerNetwork(m=(50,30), rho=0.9, n_epochs=10, n_batch=100, eta=1.28729289e-01, l=1.55476073e-05, include_train_cost=True); 

knn = KLayerNetwork(m=(50, 30), rho=0.9, n_epochs=200, n_batch=100, include_train_cost=False, batch_norm=True); 

#knn.compute_gradients_batch_norm(X_train[:,0:10], Y_train[:, 0:10])

#final, res = knn.eval_classifier(X_train[:,0:5])
#knn.j_cost_func(X_train,Y_train)
#knn.compute_gradients(X_train[:, 0:20], Y_train[:,0:20])
ans = knn.fit(X_train[:,0:100], Y_train[:,0:100], X_test, Y_test)
#ans = knn.fit(X_train[:,0:200], Y_train[:,0:200], X_test, Y_test)

#knn.compute_accuracy(X_train[:,0:100], Y_train[:,0:100])
#a = knn.eval_classifier(X_train)

#ans = knn.fit(X_train[:,0:100], Y_train[:,0:100], X_test, Y_test)

#print(knn.compute_accuracy(X_train[:,0:100],Y_train[:,0:100]))
#print(knn.compute_accuracy(X_test,Y_test))

#clf = TwoLayerNetwork(n_epochs=6, l=0.001, eta=0.01, n_batch=10, decay_rate=1, rho=0.9)
#
#train_scores, test_scores = clf.fit(X_train, Y_train, X_test, Y_test)
#
#train_scores = np.array(train_scores).reshape(clf.n_epochs,1)
#test_scores = np.array(test_scores).reshape(clf.n_epochs,1)
res = []

"""
Find the optimal lambda and eta
"""
def coarse_to_fine():
    
    global res
    
    # define 50-100 pairs of lambda and eta
    
    eta = np.array([rnd_eta_lambda(-5, -1) for i in range(75)])
    lambdas = np.array([rnd_eta_lambda(-5, -1) for i in range(75)])
    
    params = np.dstack((eta,lambdas)).reshape(75,2)
    
    res = np.empty((0,4), int)
    
    for i in range(75):
        clf = KLayerNetwork(n_epochs=3, l=lambdas[i], eta=eta[i], 
                              n_batch=100, decay_rate=1, rho=0.9, include_train_cost=False, m=(50,))
        
        clf.fit(X_train, Y_train, X_test,X_test)
        acc = clf.compute_accuracy(X_val, Y_val)
        
        p, cache = clf.eval_classifier(X_train)
        
        clf.mu = cache[3]
        clf.v = cache[4]
        
        acc2 = clf.compute_accuracy(X_val, Y_val)
        
        res = np.vstack((res, [[eta[i], lambdas[i], acc, acc2]]))
        
        print(i)
    
    # pass those to a classifier, do it in parallel.
    
    with open('outfile-coarse-2-layer.dat', 'wb') as fp:
        pickle.dump(res, fp)
    
    return res;


"""
Find the optimal lambda and eta
"""
def fine_search():
    
    # define 50-100 pairs of lambda and eta
    
    iters = 100
    
    eta = np.array([rnd_eta_lambda(-2, 0) for i in range(iters)])
    lambdas = np.array([rnd_eta_lambda(-5, -4) for i in range(iters)])
    
    res = np.empty((0,4), int)
    
    for i in range(iters):
        clf = KLayerNetwork(n_epochs=5, l=lambdas[i], eta=eta[i], m=(50,30),
                              n_batch=100, decay_rate=1, rho=0.9, include_train_cost=False)
        
        status = clf.fit(X_train, Y_train)
        
        if status == None:
            
            print("Skipping iter: " + str(i))
            
            continue;
        
        acc = clf.compute_accuracy(X_val, Y_val)
        
        p, cache = clf.eval_classifier(X_train)
        
        clf.mu = cache[3]
        clf.v = cache[4]
        
        acc2 = clf.compute_accuracy(X_val, Y_val)
        
        res = np.vstack((res, [[eta[i], lambdas[i], acc, acc2]]))
        
        print("-------------------------------------")
        print("Iter: " + str(i))
        print("Acc: " +str(acc))
        print("Acc2: " +str(acc2))
        print("Eta: " +str(eta[i]))
        print("Lambda: "+str(lambdas[i]))
        print("------------------------------------")
    
    # pass those to a classifier, do it in parallel.
    
    with open('outfile-fine-3-layer.dat', 'wb') as fp:
        pickle.dump(res, fp)
    
    return res;


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
    
print("Training set:")
#print(clf.compute_accuracy(X_train,Y_train))
print("Test set: ")
#print(clf.compute_accuracy(X_test,Y_test))


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

#plot_results(train_scores, test_scores)
#show_weights(clf)
    
#res = coarse_to_fine()