"""
@author: Simon Streicher

"""

from ranking.gaincalc import weightcalc
import logging
logging.basicConfig(level=logging.INFO)

writeoutput = True

mode = 'tests'
cases = ['weightcalc_tests']

for case in cases:
    weightcalc(mode, case, writeoutput)
