#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:18:24 2017

@author: ViniciusPantoja
"""
#%%


def plot_dataset(x, y, legend_loc='lower left'): 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(x[y==1, 0], x[y==1, 1], c='r', s=100, alpha=0.7, marker='*', label='POP 1',
               linewidth=0)
    
    ax.scatter(x[y==- 1, 0], x[y==-1, 1], c='b', s=100, alpha=0.7, marker='o', label='POP 2',
               linewidth=0)
    ax.axhline(y=0, color='k') 
    ax.axvline(x=0, color='k') 
    ax.set_xlabel('Length')
    ax.set_ylabel('Lightness') 
    ax.set_aspect('equal')
    if legend_loc: ax.legend(loc=legend_loc,fancybox=True).get_frame().set_alpha(0.5)
    ax.grid('on')
    
    
def plot_decision_boundary(network,treshold):
    import numpy
    x0v, x1v = numpy.meshgrid(numpy.linspace(-3, 3, 20), numpy.linspace(-2, 2, 20)) 

    x =numpy.hstack([x0v.reshape((-1,1)), x1v.reshape((-1,1))])
    y = network(x,treshold)
    plot_dataset(x, y, legend_loc=None)
    