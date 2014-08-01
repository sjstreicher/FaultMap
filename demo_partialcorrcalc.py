"""
@author: Simon Streicher

"""

from ranking.gaincalc import partialcorrcalc
import logging
logging.basicConfig(level=logging.INFO)

writeoutput = True

mode = 'plants'
cases = ['tennessee_eastman']

for case in cases:
    partialcorrcalc(mode, case, writeoutput)
