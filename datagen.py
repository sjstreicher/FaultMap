"""
Created on Mon Feb 24 15:27:21 2014

@author: Simon Streicher
"""

from numpy import vstack
import numpy as np


def autoreg_gen(samples, delay):
    """Generates an autoregressive set of vectors.

    A constant seed is used for testing comparison purposes.

    """

    # Define seed for initial source data
    np.random.seed(35)

    cause = np.random.randn(samples + delay)
    affected = np.zeros_like(cause)
    # Very close covariance occassionally breaks the kde estimator
    # Another small random element is added to take care of this
    # This is not expected to be a problem on any "real" data

    # Define seed for noise data
    np.random.seed(88)
    affected_random_add = np.random.rand(samples + delay)

    for i in range(delay, len(cause)):
        affected[i] = affected[i - 1] + cause[i - delay]

    affected = affected[delay:]
    cause = cause[delay:]

    affected = affected + affected_random_add[delay:]

    data = vstack([affected, cause])

    return data.T


def delay_gen(samples, delay):
    """Generates a random data vector and a pure delay companion.

    A constant seed is used for testing comparison purposes/

    """

    # Define seed for initial source data
    np.random.seed(35)

    cause = np.random.randn(samples + delay)
    affected = np.zeros_like(cause)
    # Very close covariance occassionally breaks the kde estimator
    # Another small random element is added to take care of this
    # This is not expected to be a problem on any "real" data

    # Define seed for noise data
    np.random.seed(88)
    affected_random_add = np.random.rand(samples + delay)

    for i in range(delay, len(cause)):
        affected[i] = cause[i - delay]

    affected = affected[delay:]
    cause = cause[delay:]

    affected = affected + affected_random_add[delay:]

    data = vstack([affected, cause])

    return data.T


def random_gen(samples, delay):
    """Generates two completely independent random data vectors."""

    # Generate first vector
    np.random.seed(35)
    x1 = np.random.randn(samples)

    # Generate second vector
    np.random.seed(88)
    x2 = np.random.randn(samples)

    data = vstack([x1, x2])

    return data.T
