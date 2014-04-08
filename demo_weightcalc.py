"""
@author: Simon Streicher

"""


from ranking.gaincalc import weightcalc
import logging
logging.basicConfig(level=logging.INFO)

writeoutput = True


#mode = 'test_cases'
#case = 'weightcalc_tests'
#weightcalc(mode, case, writeoutput)

#mode = 'plants'
#case = 'tennessee_eastman'
#weightcalc(mode, case, writeoutput)

mode = 'plants'
case = 'epu5_compressor'
weightcalc(mode, case, writeoutput)
