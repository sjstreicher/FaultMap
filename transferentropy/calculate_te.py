"""
Created on Mon Feb 24 14:45:08 2014

@author: Simon Streicher
"""
import getdata
import vectorselection
import te_calc


def calculate_te(delay, timelag, samples, sub_samples, mcsamples, k=1, l=1):
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

    transentropy = te_calc(x_pred, x_hist, y_hist, mcsamples)

    return transentropy
