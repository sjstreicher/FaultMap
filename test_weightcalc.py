"""
@author: Simon Streicher

"""

from ranking.gaincalc import weightcalc
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    mode = 'tests'
    case = 'weightcalc_tests'
    weightcalc(mode, case)
