"""
@author: Simon Streicher

"""

from ranking.noderank import looprank_static
from ranking.noderank import looprank_transient
import logging
logging.basicConfig(level=logging.INFO)

writeoutput = True

mode = 'test_cases'
case = 'noderank_tests'

for dummycreation in [False, True]:
    looprank_static(mode, case, dummycreation, writeoutput)

mode = 'plants'
case = 'tennessee_eastman'

for dummycreation in [False, True]:
    looprank_static(mode, case, dummycreation, writeoutput)
    looprank_transient(mode, case, dummycreation, writeoutput, True)
