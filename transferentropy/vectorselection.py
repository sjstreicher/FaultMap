"""
Created on Mon Feb 24 14:36:30 2014

@author: Simon Streicher
"""

import numpy as np


def vectorselection(data, timelag, sub_samples, k=1, l=1):
    """Generates sets of vectors for calculating transfer entropy.

    For notation references see Shu2013.

    Takes into account the time lag (number of samples between vectors of the
    same variable).

    In this application the prediction horizon (h) is set to equal
    to the time lag.

    The first vector in the data array should be the samples of the variable
    to be predicted (x) while the second vector should be sampled of the vector
    used to make the prediction (y).

    sub_samples is the amount of samples in the dataset used to calculate the
    transfer entropy between two vectors.
    The required number of samples is extracted from the end of the vector.
    If the vector is longer than the number of samples specified plus the
    desired time lag then the remained of the data will be discarded.
    sub_samples <= samples


    k refers to the dimension of the historical data to be predicted (x)

    l refers to the dimension of the historical data used
    to do the prediction (y)

    """
    _, sample_n = data.shape
    x_pred = data[0, sample_n-sub_samples-1:-1]

    x_hist = np.zeros((k, sub_samples))
    y_hist = np.zeros((l, sub_samples))

    for n in range(1, (k+1)):
        # Original form according to Bauer (2007)
#        x_hist[n-1, :] = data[0, ((sample_n - samples) - timelag * n):
#                               (sample_n - timelag * n)]
        # Modified form according to Shu & Zhao (2013)
        x_hist[n-1, :] = data[0, ((sample_n - sub_samples) - timelag *
                                  (n-1) - 2):(sample_n - timelag * (n-1) - 2)]
    for m in range(1, (l+1)):
        y_hist[m-1:, :] = data[1, ((sample_n - sub_samples) -
                               timelag * (m) - 1):
                               (sample_n - timelag * (m) - 1)]

    return x_pred, x_hist, y_hist
