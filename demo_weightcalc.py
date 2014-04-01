"""
@author: Simon Streicher

"""


from ranking.gaincalc import weightcalc

writeoutput = True


mode = 'test_cases'
case = 'weightcalc_tests'
weightcalc(mode, case, writeoutput)

mode = 'plants'
case = 'tennessee_eastman'
weightcalc(mode, case, writeoutput)
