"""
@author: Simon Streicher

"""

from ranking.noderank import looprank_static

if __name__ == '__main__':
    mode = 'test_cases'
    case = 'noderank_tests'
    dummycreation = False
    looprank_static(mode, case, dummycreation, False)
