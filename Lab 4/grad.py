# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 21:25:05 2018

@author: Artem Los
"""

class Grad:
    
    def __init__(self, dLdh, dLdo, dLdV, dLdW, dLdU, dLdb, dLdc):
        
        self.dLdh = dLdh
        self.dLdo = dLdo
        
        self.dLdV = dLdV
        self.dLdW = dLdW
        self.dLdU = dLdU
        
        self.dLdb = dLdb
        self.dLdc = dLdc