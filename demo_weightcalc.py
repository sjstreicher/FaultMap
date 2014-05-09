"""
@author: Simon Streicher

"""

from ranking.gaincalc import weightcalc, partialcorrcalc
import logging
logging.basicConfig(level=logging.INFO)

writeoutput = True
sigtest = True

mode = 'plants'
cases = ['propylene_compressor']

for case in cases:
    weightcalc(mode, case, sigtest, writeoutput)
#    partialcorrcalc(mode, case, writeoutput)
