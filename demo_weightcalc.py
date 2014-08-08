"""
@author: Simon Streicher

"""

from ranking.gaincalc import weightcalc
import logging
import timeit
logging.basicConfig(level=logging.INFO)

writeoutput = True
sigtest = False

mode = 'plants'
case = ['filters']


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

wrapped = wrapper(weightcalc, mode, case[0], sigtest, writeoutput)

print timeit.timeit(wrapped, number=1)
