# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:23:44 2018

@author: Artem Los
"""

import numpy as np
from helpers import *
import matplotlib.pyplot as plt

dataset = unpickle('Datasets/data_batch_1')

images = dataset[b'data']

show_random_images(images)