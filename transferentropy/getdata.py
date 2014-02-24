"""
Created on Mon Feb 24 14:34:58 2014

@author: Simon Streicher
"""
import autogen


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
