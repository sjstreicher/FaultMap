"""
Created on Thu Feb 06 10:50:28 2014

@author: Simon Streicher
"""

from numpy import loadtxt, vstack
import numpy as np
from autoregres_gen import autogen
from transfer_entropy import vectorselection, te


def getdata(samples, delay):
    """Get dataset for testing.

    Select to generate each run or import an existing dataset.

    """

    # Generate autoregressive delayed data vectors internally
    data = autogen(samples, delay)

    # Alternatively, import data from file
#    autoregx = loadtxt('autoregx_data.csv')
#    autoregy = loadtxt('autoregy_data.csv')

    return data


def calculate_te(delay, timelag, samples, sub_samples, ampbins, k=1, l=1):
    """Calculates the transfer entropy for a specific timelag (equal to
    prediction horison) for a set of autoregressive data.

    sub_samples is the amount of samples in the dataset used to calculate the
    transfer entropy between two vectors (taken from the end of the dataset).
    sub_samples <= samples

    Currently only supports k = 1; l = 1;

    You can search through a set of timelags in an attempt to identify the
    original delay.
    The transfer entropy should have a maximum value when timelag = delay
    used to generate the autoregressive dataset.

    """
    # Get autoregressive datasets
    data = getdata(samples, delay)

    [x_pred, x_hist, y_hist] = vectorselection(data, timelag,
                                               sub_samples, k, l)

    tentropy = te(x_pred, x_hist, y_hist, ampbins)

    return tentropy

# Test code
# Delay = 5, Timelag = 4
tentropy1 = calculate_te(5, 4, 1000, 500, 10)
# Delay = 5, Timelag = 5
tentropy2 = calculate_te(5, 5, 1000, 500, 10)
# Delay = 5, Timelag = 6
tentropy3 = calculate_te(5, 6, 1000, 500, 10)


#samples = 30
#delay = 5
#
#data = autogen(samples, delay)
#x = data[0]
#y = data[1]
#
#x_diff = np.zeros_like(x[delay-1:-1])
#for i in range(delay, len(data[0])):
#    x_diff[i - delay] = x[i] - x[i - 1]