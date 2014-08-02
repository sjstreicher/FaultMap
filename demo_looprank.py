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
m = 0.15

mode = 'plants'
cases = ['propylene_compressor']

for case in cases:
    for dummycreation in [False, True]:
        looprank_static(mode, case, dummycreation, writeoutput,
                        m, alpha)
