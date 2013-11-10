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
import scipy
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

#autoregx = importcsv('autoregx_data.csv')
#autoregy = importcsv('autoregy_data.csv')

#data = np.vstack([autoregx, autoregy])
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
        # Original form according to Bauer (2007)
#        x_hist[n-1, :] = data[0, ((sample_n - samples) - timelag * n):(sample_n - timelag * n)]
        # Modified form according to Shu & Zhao (2013)
        x_hist[n-1, :] = data[0, ((sample_n - samples) - timelag * (n-1) -1):(sample_n - timelag * (n-1) -1)]
    for m in range(1, (l+1)):
        y_hist[m-1:, :] = data[1, ((sample_n - samples) - timelag * m):(sample_n - timelag * m)]

#    for n in range(1, (k+1)):
#        x_hist = data[0, ((sample_n - samples) - timelag * n):(sample_n - timelag * n)]
#    for m in range(1, (l+1)):
#        y_hist = data[1, ((sample_n - samples) - timelag * m):(sample_n - timelag * m)]

    return x_pred, x_hist, y_hist

#[x_pred, x_hist, y_hist] = vectorselection(data, 10, 3000, 1, 1)


def te(x_pred, x_hist, y_hist, ampbins):
    """Calculates the transfer entropy between two variables from a set of
    vectors already calculated.
    ampbins is the number of amplitude bins to use over each variable
    First do an example for the case of k = l = 1
    Go smart about the for summing loops to allow for a general case
    """

    # Divide the range of each variable into amplitude bins to sum over
    x_pred_min = x_pred.min()
    x_pred_max = x_pred.max()
    x_hist_min = x_hist.min()
    x_hist_max = x_hist.max()
    y_hist_min = y_hist.min()
    y_hist_max = y_hist.max()

    x_pred_space = np.linspace(x_pred_min, x_pred_max, ampbins)
    x_hist_space = np.linspace(x_hist_min, x_hist_max, ampbins)
    y_hist_space = np.linspace(y_hist_min, y_hist_max, ampbins)

    x_pred_diff = x_pred_space[1] - x_pred_space[0]
    x_hist_diff = x_hist_space[1] - x_hist_space[0]
    y_hist_diff = y_hist_space[1] - y_hist_space[0]

    # Calculate PDFs for all combinations required
    [pdf_1, pdf_2, pdf_3, pdf_4] = pdfcalcs(x_pred, x_hist, y_hist)

    # Consecutive sums

    # TODO: Make sure Riemann sum diff elements is handled correctly

    tesum = 0
    delement = x_pred_diff * x_hist_diff * y_hist_diff
    print delement
    for s1 in x_pred_space:
#        print 's1', s1
        for s2 in x_hist_space:
#            print 's2', s2
            for s3 in y_hist_space:
#                print 's3', s3
                sum_element = tecalc(pdf_1, pdf_2, pdf_3, pdf_4, s1, s2, s3)
                tesum = tesum + sum_element
    te = tesum * delement

    # Using local sums
    # (It does give the same result)

#    sums3 = 0
#    sums2 = 0
#    sums1 = 0
#    for s1 in x_pred_space:
#        print s1
#        sums2 = 0
#        for s2 in x_hist_space:
##            print s2
#            sums3 = 0
#            for s3 in y_hist_space:
#                sum_element = tecalc(pdf_1, pdf_2, pdf_3, pdf_4, s1, s2, s3)
#                sums3 = sums3 + sum_element
#            sums2 = sums2 + sums3 * y_hist_diff
#        sums1 = sums1 + sums2 * x_hist_diff
#        te = sums1 * x_pred_diff

    return te


def pdfcalcs(x_pred, x_hist, y_hist):
    """Calculates the PDFs required to calculate transfer entropy"""

    # Get dimensions of vectors
#    k = np.size(x_hist[:, 1])
#    l = np.size(y_hist[:, 1])

    # Currently only works for k = 1; l = 1
    # TODO: Generalize for k and l

    # Calculate p(x_{i+h}, x_i, y_i)
    data_1 = np.vstack([x_pred, x_hist[0, :], y_hist[0, :]])
    pdf_1 = stats.gaussian_kde(data_1, 'silverman')

    # Calculate p(x_i, y_i)
    data_2 = np.vstack([x_hist[0, :], y_hist[0, :]])
    pdf_2 = stats.gaussian_kde(data_2, 'silverman')

    # Calculate p(x_{i+h}, x_i)
    data_3 = np.vstack([x_pred, x_hist[0, :]])
    pdf_3 = stats.gaussian_kde(data_3, 'silverman')

    # Calculate p(x_i)
    data_4 = x_hist[0, :]
    pdf_4 = stats.gaussian_kde(data_4, 'silverman')

    return pdf_1, pdf_2, pdf_3, pdf_4


def tecalc(pdf_1, pdf_2, pdf_3, pdf_4, x_pred_val, x_hist_val, y_hist_val):
    """Calculate elements for summation for a specific set of coordinates"""

    # Need to find a proper way to correct for cases when PDFs return 0

    # Function evaluations
    term1 = pdf_1([x_pred_val, x_hist_val, y_hist_val])
    term2 = pdf_2([x_hist_val, y_hist_val])
    term3 = pdf_3([x_pred_val, x_hist_val])
    term4 = pdf_4([x_hist_val])
#    if term1 == 0 or term2 == 0 or term3 == 0 or term4 == 0:
#        sum_element = 0
#        print term1, term2, term3, term4
#
#    else:
#        logterm_num = (term1 / term2)
#        logterm_den = (term3 / term4)
#        coeff = term1
#        sum_element = coeff * np.log(logterm_num / logterm_den)
#        print np.log(logterm_num / logterm_den)

    logterm_num = (term1 / term2)
    logterm_den = (term3 / term4)
    coeff = term1
    sum_element = coeff * np.log(logterm_num / logterm_den)

    return sum_element


"""Testing commands"""

[x_pred, x_hist, y_hist] = vectorselection(data, 60, 3000, 1, 1)

TE = te(x_pred, x_hist, y_hist, 20)
print 'TE: ', float(TE)

[pdf_1, pdf_2, pdf_3, pdf_4] = pdfcalcs(x_pred, x_hist, y_hist)


#def x_max(x, x_pred, x_hist, y_hist):
#    # There must be a better way than calculating PDFs in each iteration
#    [pdf_1, pdf_2, pdf_3, pdf_4] = pdfcalcs(x_pred, x_hist, y_hist)
#    return -pdf_4([x])
#
#max_x = scipy.optimize.fmin(x_max, 0, args=(x_pred, x_hist, y_hist))



#[te, sums1, x_pred_space] = te(x_pred, x_hist, y_hist, 100)

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