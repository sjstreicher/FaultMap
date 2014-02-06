"""
Created on Fri Nov 08 15:37:47 2013

@author: Simon Streicher
"""

from numpy import vstack
import numpy as np


def autogen(samples, delay):
    """Generate an autoregressive set of vectors."""

    source = np.random.randn(samples + delay + 1)
    pred = np.zeros_like(source)

    for i in range(delay, len(source)):
        pred[i] = pred[i - 1] + source[i - delay]

    pred = pred[delay:-1]
    source = source[delay:-1]

    data = vstack([pred, source])

    return data

# Some test code
#SAMPLES = 30
#DELAY = 5
#TEST_DATA = autogen(SAMPLES, DELAY)
#PRED = TEST_DATA[0]
#SOURCE = TEST_DATA[1]
#PRED_DIFF = np.zeros_like(PRED[DELAY-1:-1])
#for i in range(DELAY, len(TEST_DATA[0])):
#    PRED_DIFF[i - DELAY] = PRED[i] - PRED[i - 1]
# Test that the first few PRED_DIFF entries are equal to that of SOURCE