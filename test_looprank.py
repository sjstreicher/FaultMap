"""
@author: Simon Streicher

"""

from ranking.noderank import looprank

if __name__ == '__main__':
    mode = 'tests'
    case = 'noderank_tests'
    writeoutput = False

    alpha = 0.5
    m = 0.15
    
	# looprank function broken for test cases at the moment (23/09/2014)
	# will correct once the new structure has been determined
	
    # Run without dummy creation
    # dummycreation = False
    # looprank(mode, case, dummycreation, writeoutput, alpha, m)

    # Run with dummy creation
    # dummycreation = True
    # looprank(mode, case, dummycreation, writeoutput, alpha, m)
