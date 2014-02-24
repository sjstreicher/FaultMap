"""
Created on Mon Feb 24 15:27:21 2014

@author: Simon Streicher
"""

from numpy import vstack
import numpy as np


def autogen(samples, delay):
    """Generate an autoregressive set of vectors."""

    source = np.random.randn(samples + delay + 1)
    pred = np.zeros_like(source)
    # Very close covariance occassionally breaks the kde estimator
    # Another small random element is added to take care of this
    # This is not expected to be a problem on any "real" data
    pred_random_add = np.random.rand(samples + delay + 1)

    for i in range(delay, len(source)):
        pred[i] = pred[i - 1] + source[i - delay]

    pred = pred[delay:-1]
    source = source[delay:-1]

    pred = pred + pred_random_add[delay:-1]

    data = vstack([pred, source])

    return data


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
