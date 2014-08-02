"""
@author: Simon Streicher

"""

from ranking.noderank import looprank_static
import logging
logging.basicConfig(level=logging.INFO)

writeoutput = True

# Ranking parameters
# This is now the only place were these parameters are defined to avoid
# hardcoded overwriting
alpha = 0.5
m = 0.9999

mode = 'plants'
cases = ['presentation_sample']

for case in cases:
    for dummycreation in [False]:
        looprank_static(mode, case, dummycreation, writeoutput,
                        m, alpha)
