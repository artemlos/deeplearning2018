# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 09:36:41 2018

@author: Artem Los
"""

import numpy as np
from helpers import *
import matplotlib.pyplot as plt

def part2():
    three_layer_with_br = unpickle('2-3-layer-with-br-train-test.dat')
    three_layer_without_br = unpickle('2-3-layer-without-br-train-test.dat')
    
    plt.figure(0)
    plt.title("Three layer network with batch norm")
    plt.plot(three_layer_with_br[0], label='Train')
    plt.plot(three_layer_with_br[1], label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend()
    plt.savefig('part2-with-bn.png')
    
    plt.figure(1)
    plt.title("Three layer network without batch norm")
    plt.plot(three_layer_without_br[0], label='Train')
    plt.plot(three_layer_without_br[1], label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend()
    plt.savefig('part2-without-bn.png')
    
    
def part4():
    
    nobn_small = unpickle('4-2layer-nobn-smalleta.dat')
    nobn_medium = unpickle('4-2layer-nobn-mediumeta.dat')
    nobn_high = unpickle('4-2layer-nobn-higheta.dat')
    bn_high = unpickle('4-2layer-bn-higheta.dat')
    bn_medium = unpickle('4-2layer-bn-mediumeta.dat')
    bn_small = unpickle('4-2layer-bn-smalleta.dat')
    
    plt.figure(0)
    plt.title("Two layer network, without batch norm, high eta")
    plt.plot(nobn_high[0], label='Train')
    plt.plot(nobn_high[1], label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend()
    plt.savefig('part4-nobn_high.png')
    
    plt.figure(1)
    plt.title("Two layer network, without batch norm, medium eta")
    plt.plot(nobn_medium[0], label='Train')
    plt.plot(nobn_medium[1], label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend()
    plt.savefig('part4-nobn_medium.png')
    
    plt.figure(2)
    plt.title("Two layer network, without batch norm, small eta")
    plt.plot(nobn_small[0], label='Train')
    plt.plot(nobn_small[1], label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend()
    plt.savefig('part4-nobn_small.png')
    
    
    plt.figure(3)
    plt.title("Two layer network, with batch norm, high eta")
    plt.plot(bn_high[0], label='Train')
    plt.plot(bn_high[1], label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend()
    plt.savefig('part4-bn_high.png')
    
    plt.figure(4)
    plt.title("Two layer network, with batch norm, medium eta")
    plt.plot(bn_medium[0], label='Train')
    plt.plot(bn_medium[1], label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend()
    plt.savefig('part4-bn_medium.png')
    
    plt.figure(5)
    plt.title("Two layer network, with batch norm, small eta")
    plt.plot(bn_small[0], label='Train')
    plt.plot(bn_small[1], label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend()
    plt.savefig('part4-bn_small.png')
    
part2(); 

def analyse_coarse():
    # analyse the file
    data = unpickle('outfile-coarse-2-layer.dat')
    data2 = unpickle('outfile-fine-2-layer-diff-searchrange-iter.dat')
    
    data = np.vstack((data,data2))
    # extract those indicies that are above 0.38 in accuracy
    filtered = np.where(data[:,3]>=0.42)
    
    print(data[filtered])
    
def analyse_coarse3():
    # analyse the file
    data = unpickle('outfile-coarse-3-layer.dat')
    
    # extract those indicies that are above 0.38 in accuracy
    filtered = np.where(data[:,3]>=0.3)
    
    print(data[filtered])
    
    

def analyse_fine():
    data = unpickle('Results/outfile-fine-nodecay-7iter.dat')
    
    data2 = unpickle('Results/outfile-fine-nodecay-7iter.dat')
    
    data3 = unpickle('Results/outfile-fine-iter7-nodecay-100.dat')
    
    data4 = unpickle('Results/outfile-fine-10iter-100-nodecay.dat')
    
    data5 = unpickle('Results/outfile-fine-new-nodecay-2-1-4-3-10iter.dat')
    
    data6 = unpickle('Results/outfile-fine-new-nodecay-2-1-5-4-10iter.dat')
    
    data = np.vstack((data, data2, data3,data4,data5,data6))
    
    filtered = np.where(data[:,2]>=0.433)
    
    print(data[filtered])
    
    return