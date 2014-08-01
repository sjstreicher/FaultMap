"""
@author: Simon Streicher

"""


from ranking.gaincalc import weightcalc

if __name__ == '__main__':
    mode = 'tests'
    case = 'weightcalc_tests'
    writeoutput = False
    sigtest = True
    weightcalc(mode, case, sigtest, writeoutput)
