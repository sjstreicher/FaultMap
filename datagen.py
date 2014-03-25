"""
Created on Mon Feb 24 15:27:21 2014

@author: Simon Streicher
"""

from numpy import vstack
import numpy as np


def autoreg_gen(samples, delay):
    """Generate an autoregressive set of vectors.

    A constant seed is used for testing comparison purposes.

    """

    # Define seed for initial source data
    np.random.seed(35)

    source = np.random.randn(samples + delay + 1)
    pred = np.zeros_like(source)
    # Very close covariance occassionally breaks the kde estimator
    # Another small random element is added to take care of this
    # This is not expected to be a problem on any "real" data

    # Define seed for noise data
    np.random.seed(42)
    pred_random_add = np.random.rand(samples + delay + 1)

    for i in range(delay, len(source)):
        pred[i] = pred[i - 1] + source[i - delay]

    pred = pred[delay:-1]
    source = source[delay:-1]

    pred = pred + pred_random_add[delay:-1]

    data = vstack([pred, source])

    return data.T
