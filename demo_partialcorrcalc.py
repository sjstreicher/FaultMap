"""
@author: Simon Streicher

"""

from ranking.gaincalc import partialcorrcalc
import logging
logging.basicConfig(level=logging.INFO)

writeoutput = True

runs = ['u238_alcoholrecovery']

if 'weightcalc_tests' in runs:
    mode = 'test_cases'
    case = 'weightcalc_tests'
    partialcorrcalc(mode, case, writeoutput)

if 'tennessee_eastman' in runs:
    mode = 'plants'
    case = 'tennessee_eastman'
    partialcorrcalc(mode, case, writeoutput)

if 'epu5_compressor' in runs:
    mode = 'plants'
    case = 'epu5_compressor'
    partialcorrcalc(mode, case, writeoutput)

if 'u238_alcoholrecovery' in runs:
    mode = 'plants'
    case = 'u238_alcoholrecovery'
    partialcorrcalc(mode, case, writeoutput)
