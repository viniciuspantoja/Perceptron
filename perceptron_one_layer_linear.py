#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:31:11 2017

@author: ViniciusPantoja
"""

#%%
import numpy as np

class Perceptron(object):
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y, treshold):
        self.w_ = np.zeros(1 + X.shape[1])
        
        self.errors = []
        
        for _ in range (self.n_iter):
            
            errors = 0
            for xi, target in zip(X,y):
                
                update = self.eta*(target - self.predict(xi, treshold))

                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update !=0.0)

                
                
                
                
            self.errors.append(errors)
            
        return self

    def net_input(self, x):
        a = np.dot(x, self.w_[1:]) + self.w_[0]
        return a
    
    def predict(self,x, treshold):
        a = np.where(self.net_input(x) >= treshold, 1, -1)

        return a
    
if __name__ == '__main__':
    Perceptron = Perceptron()
    Perceptron.run()