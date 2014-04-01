"""
@author: Simon Streicher

"""

from ranking.noderank import looprank_static

if __name__ == '__main__':
    mode = 'test_cases'
    case = 'noderank_tests'
    writeoutput = False

    # Run without dummy creation
    dummycreation = False
    looprank_static(mode, case, dummycreation, writeoutput)

    # Run with dummy creation
    dummycreation = True
    looprank_static(mode, case, dummycreation, writeoutput)
