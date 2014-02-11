"""
Created on Mon Nov 11 17:35:30 2013

@author: Simon Streicher
"""

from numpy import array, transpose, zeros, hstack
import csv
import numpy as np
from numpy import loadtxt
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from random import random
from math import isnan

#def createCorrelationGainMatrix(statesloc):
"""This method strives to calculate the local gains in terms of the correlation
between the variables. It uses the partial correlation method (Pearson's
correlation)."""

n = 53

# Not sure why this does not work
#dataholder = loadtxt('data.csv')

fromfile = csv.reader(open('data.csv'), delimiter = ',')
dataholder = []

for row in fromfile:
    rowfix = row
    for element in rowfix:
        dataholder.append(float(element))

inputdata = np.reshape(dataholder, (-1, n))

correlationmatrix = array(np.empty((n, n)))

for i in range(n):
#    print 'i = ', i
    for j in range(n):
#        print 'j = ', j
        temp = pearsonr(inputdata[:, i], inputdata[:, j])[0]
        if isnan(temp):
#            print 'Here!!'
            correlationmatrix[i,j] = random()*0.01
        else:
            correlationmatrix[i,j] = temp

    P = np.linalg.inv(correlationmatrix)
    partialcorrelationmatrix = array(np.empty((n, n)))

    for i in range(n):
        for j in range(n):
            if 1 == 1:
                temp = -1*P[i,j]/( abs( (P[i,i]*P[j,j]) )**0.5 )
                partialcorrelationmatrix[i,j] = temp
            else:
                partialcorrelationmatrix[i,j] = 0

#    return correlationmatrix

# Testing script