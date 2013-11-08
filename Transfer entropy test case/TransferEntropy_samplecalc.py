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


def importcsv(file):
    """Imports csv file and returns values in array"""
    fromfile = csv.reader(open(file), delimiter=' ')
    temp = []
    for row in fromfile:
        temp.append(float(row[0]))
    temp = array(temp)
    return temp

original = importcsv('original_data.csv')
puredelay = importcsv('puredelay_data.csv')

data = np.vstack([original, puredelay])
kernel = stats.gaussian_kde(data, 'silverman')

#xmin = original.min()
#xmax = original.max()
#ymin = puredelay.min()
#ymax = puredelay.max()
#X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
#positions = np.vstack([X.ravel(), Y.ravel()])
#Z = np.reshape(kernel(positions).T, X.shape)
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
#          extent=[xmin, xmax, ymin, ymax])
ax.plot(original, puredelay, 'k.', markersize=2)
#ax.set_xlim([xmin, xmax])
#ax.set_ylim([ymin, ymax])
plt.show()