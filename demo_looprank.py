"""
@author: Simon Streicher

"""

from ranking.noderank import looprank_static
import logging
logging.basicConfig(level=logging.INFO)

writeoutput = True

runs = ['u238_alcoholrecovery']

if 'noderank_tests' in runs:
    mode = 'test_cases'
    case = 'noderank_tests'
    for dummycreation in [False, True]:
        looprank_static(mode, case, dummycreation, writeoutput)

if 'tennessee_eastman' in runs:
    mode = 'plants'
    case = 'tennessee_eastman'
    for dummycreation in [False, True]:
        looprank_static(mode, case, dummycreation, writeoutput)

if 'epu5_compressor' in runs:
    mode = 'plants'
    case = 'epu5_compressor'
    for dummycreation in [False, True]:
        looprank_static(mode, case, dummycreation, writeoutput)

if 'u238_alcoholrecovery' in runs:
    mode = 'plants'
    case = 'u238_alcoholrecovery'
    for dummycreation in [False, True]:
        looprank_static(mode, case, dummycreation, writeoutput)
