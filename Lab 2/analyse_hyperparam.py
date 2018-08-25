# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 09:36:41 2018

@author: Artem Los
"""

import numpy as np
from helpers import *


def analyse_coarse():
    # analyse the file
    data = unpickle('Results/eta-lambda-3-epochs.dat')
    
    # extract those indicies that are above 0.38 in accuracy
    filtered = np.where(data[:,2]>=0.386)
    
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

analyse_fine()