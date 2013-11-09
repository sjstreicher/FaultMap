# -*- coding: utf-8 -*-
"""
Created on Fri Nov 08 15:37:47 2013

@author: s13071832
"""

"""Calculates transfer entropy between sample signals"""

"""Import classes and modules"""
import csv
from numpy import array
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

#from numpy import transpose, zeros, hstack
#import networkx as nx
#from scipy.stats.stats import pearsonr
#from random import random
#from math import isnan


def importcsv(filename):
    """Imports csv file and returns values in array"""
    fromfile = csv.reader(open(filename), delimiter=' ')
    temp = []
    for row in fromfile:
        temp.append(float(row[0]))
    temp = array(temp)
    return temp

original = importcsv('original_data.csv')
puredelay = importcsv('puredelay_data.csv')
delayedtf = importcsv('delayedtf_data.csv')

data = np.vstack([puredelay, original])
kernel = stats.gaussian_kde(data, 'silverman')

#def multivarPDF(data):

def vectorselection(data, timelag, samples, k, l):
    """Generates sets of vectors for calculating transfer entropy.
    Takes into account the time lag (number of samples between vectors of the
    same variable).
    In this application the prediction horizon is set to equal to the time lag.
    The first vector in the data array should be the samples of the variable
    to be predicted while the second vector should be sampled of the vector
    used to make the prediction.
    The required number of samples is extracted from the end of the vector.
    If the vector is longer than the number of samples specified plus the
    desired time lag then the remained of the data will be discarded.
    k refers to the dimension of the historical data to be predicted
    l refers to the dimension of the historical data used to do the prediction
    """
    sample_n = np.size(data[1, :])
    x_pred = data[0, (sample_n - samples):sample_n]
    x_pred = data[0, (sample_n - samples):sample_n]
    
    x_hist = np.zeros((k, samples))
    y_hist = np.zeros((l, samples))
    
    for n in range(1, (k+1)):
        x_hist[n-1, :] = data[0, ((sample_n - samples) - timelag * n):(sample_n - timelag * n)]
    for m in range(1, (l+1)):
        y_hist[m-1:, :] = data[1, ((sample_n - samples) - timelag * m):(sample_n - timelag * m)]
    
#    for n in range(1, (k+1)):
#        x_hist = data[0, ((sample_n - samples) - timelag * n):(sample_n - timelag * n)]
#    for m in range(1, (l+1)):
#        y_hist = data[1, ((sample_n - samples) - timelag * m):(sample_n - timelag * m)]

    return x_pred, x_hist, y_hist

#[x_pred, x_hist, y_hist] = vectorselection(data, 10, 3000, 1, 1)

def te(x_pred, x_hist, y_hist):
    """Calculates the transfer entropy between two variables from a set of
    vectors already calculated.
    """
    
    """Get dimensions of vectors"""
    k = np.size(x_hist[:, 1])
    l = np.size(y_hist[:, 1])
    
    
    
    """Set up consecutive sums"""
    
    
    
    te = 1
    return te

#xmin = original.min()
#xmax = original.max()
#ymin = puredelay.min()
#ymax = puredelay.max()
#X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
#positions = np.vstack([X.ravel(), Y.ravel()])
#Z = np.reshape(kernel(positions).T, X.shape)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
#          extent=[xmin, xmax, ymin, ymax])
#ax.plot(original, puredelay, 'k.', markersize=2)
#ax.set_xlim([xmin, xmax])
#ax.set_ylim([ymin, ymax])
#plt.show()