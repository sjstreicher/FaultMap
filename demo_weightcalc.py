"""
@author: Simon Streicher

"""

from ranking.gaincalc import weightcalc
import logging
logging.basicConfig(level=logging.INFO)

writeoutput = True
sigtest = False

mode = 'plants'
cases = ['propylene_compressor']

for case in cases:
    weightcalc(mode, case, sigtest, writeoutput)
