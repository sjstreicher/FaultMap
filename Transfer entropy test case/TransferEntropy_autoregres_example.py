# -*- coding: utf-8 -*-
"""
Created on Fri Nov 08 15:37:47 2013

@author: s13071832
"""

"""Calculates transfer entropy between sample signals"""

"""Import classes and modules"""
import csv
from numpy import array
from random import gauss
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

# TODO: Generate this data in Python itself
autoregx = importcsv('autoregx_data_excelgen.csv')
autoregy = importcsv('autoregy_data_excelgen.csv')

data = np.vstack([autoregx, autoregy])


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
#        x_hist[n-1, :] = data[0, ((sample_n - samples) - timelag * n):
#                               (sample_n - timelag * n)]
        # Modified form according to Shu & Zhao (2013)
        x_hist[n-1, :] = data[0, ((sample_n - samples) - timelag * (n-1) - 1):
                              (sample_n - timelag * (n-1) - 1)]
    for m in range(1, (l+1)):
        y_hist[m-1:, :] = data[1, ((sample_n - samples) - timelag * m):
                               (sample_n - timelag * m)]

    return x_pred, x_hist, y_hist


def te(x_pred, x_hist, y_hist, ampbins):
    """Calculates the transfer entropy between two variables from a set of
    vectors already calculated.
    ampbins is the number of amplitude bins to use over each variable
    """

    # This only works for k = l = 1
    # TODO: Implement summing loop for a general case

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
#    print delement
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
    # Most of the PDF issues are associated with the x_hist values being
    # very similar to the x_pred values
    # This will not be an issue if using k = 0 and l = 2 as in the work
    # of Bauer (2005)
    # Some very small negative values are sometimes returned - find out if
    # this is realistic

    # Function evaluations
    term1 = pdf_1([x_pred_val, x_hist_val, y_hist_val])
    term2 = pdf_2([x_hist_val, y_hist_val])
    term3 = pdf_3([x_pred_val, x_hist_val])
    term4 = pdf_4([x_hist_val])
#    print term1, term2, term3, term4

    # Temporary solution: assigns small values to evaluations
    # below a certain threshold

    if term1 < 1e-300:
        term1 = 1e-300

    if term2 < 1e-300:
        term2 = 1e-300

    if term3 < 1e-300:
        term3 = 1e-300

    if term4 < 1e-300:
        term4 = 1e-300

    logterm_num = (term1 / term2)
    logterm_den = (term3 / term4)
    coeff = term1
    sum_element = coeff * np.log(logterm_num / logterm_den)

    # Some negative sum elements due to logterm_num being smaller than
    # logterm_den, not sure if this is realistic

    # Eliminate negative sum terms

    if sum_element < 0:
        sum_element = 0
#        print logterm_num, logterm_den

#    print sum_element

    return sum_element

"""Testing commands"""

# Test for special case of h = 5 to see source of problem
# Expected source: covariance in kernel function

# Proposed solution: add a small bit of noise to x values in order to
# break covariance issue

autoregx_noise = importcsv('autoregx_data_noise_excelgen.csv')

data_noise = np.vstack([autoregx_noise, autoregy])

#[x_pred, x_hist, y_hist] = vectorselection(data_noise, 5, 2500, 1, 1)
#TE_5 = te(x_pred, x_hist, y_hist, 20)

#print 'The TE at h = 5 evaluates to: ', TE_5


def grandcalc(data, lag, samples, k, l, ampbins):
    [x_pred, x_hist, y_hist] = vectorselection(data, lag, samples, k, l)
    TE = te(x_pred, x_hist, y_hist, ampbins)
    return TE

# Generate list of TEs for h = 1 to h = 20

print 'The current lag in the data is 5.'
print 'Expected peaks at multiples of 5.'

TE_list = np.zeros(20)
for m in range(0, 20):
    TE_list[m] = grandcalc(data_noise, (m+1), 2500, 1, 1, 20)
    print 'TE for h = ', (m+1), ' is: ', float(TE_list[m])


# Generate PDFs to test in console
#[pdf_1, pdf_2, pdf_3, pdf_4] = pdfcalcs(x_pred, x_hist, y_hist)

# Example of finding the location of the highest probability
#def x_max(x, x_pred, x_hist, y_hist):
#    # There must be a better way than calculating PDFs in each iteration
#    [pdf_1, pdf_2, pdf_3, pdf_4] = pdfcalcs(x_pred, x_hist, y_hist)
#    return -pdf_4([x])
#
#max_x = scipy.optimize.fmin(x_max, 0, args=(x_pred, x_hist, y_hist))