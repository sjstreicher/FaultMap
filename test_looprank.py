"""
@author: Simon Streicher

"""

from ranking.noderank import looprank_static

if __name__ == '__main__':
    mode = 'test_cases'
    case = 'noderank_tests'
    writeoutput = False

    alpha = 0.5
    m = 0.15

    # Run without dummy creation
    dummycreation = False
    looprank_static(mode, case, dummycreation, writeoutput, alpha, m)

    # Run with dummy creation
    dummycreation = True
    looprank_static(mode, case, dummycreation, writeoutput, alpha, m)
