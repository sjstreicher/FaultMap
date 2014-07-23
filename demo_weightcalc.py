"""
@author: Simon Streicher

"""

from ranking.gaincalc import weightcalc
import logging
logging.basicConfig(level=logging.INFO)

writeoutput = True
sigtest = True

mode = 'plants'
cases = ['tennessee_eastman']

for case in cases:
    weightcalc(mode, case, sigtest, writeoutput)
