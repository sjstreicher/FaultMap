"""
@author: Simon Streicher

"""

from ranking.gaincalc import weightcalc
import logging
logging.basicConfig(level=logging.INFO)

writeoutput = True

runs = ['u238_alcoholrecovery']

if 'weightcalc_tests' in runs:
    mode = 'test_cases'
    case = 'weightcalc_tests'
    weightcalc(mode, case, writeoutput)

if 'tennessee_eastman' in runs:
    mode = 'plants'
    case = 'tennessee_eastman'
    weightcalc(mode, case, writeoutput)

if 'epu5_compressor' in runs:
    mode = 'plants'
    case = 'epu5_compressor'
    weightcalc(mode, case, writeoutput)

if 'u238_alcoholrecovery' in runs:
    mode = 'plants'
    case = 'u238_alcoholrecovery'
    weightcalc(mode, case, writeoutput)
