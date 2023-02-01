# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 13:06:31 2022

@author: razze
"""

class train_parameters(object):
    def __init__(self, max_outer_iteration, max_inner_iteration, learning_rate):
        self.max_outer_iteration = max_outer_iteration
        self.max_inner_iteration = max_inner_iteration
        self.learning_rate = learning_rate