from test import networkgen

import numpy as np

from faultmap import data_processing, noderank

connections, gainmatrix, variables, testgraph = networkgen.fullconn_random()
biasvector = np.ones(len(variables))
dummyweight = 10
dummies = False

rank_method = "eigenvector"


class NodeRankdata(object):
    def __init__(self, m, alpha):
        self.m = m
        self.alpha = alpha


noderankdata = NodeRankdata(0.999, 0.1)

(
    backwardconnection,
    backwardgain,
    backwardvariablelist,
    backwardbias,
) = data_processing.rank_backward(
    variables, gainmatrix, connections, biasvector, dummyweight, dummies
)

connections = [backwardconnection]
variables = [backwardvariablelist]
gains = [np.array(backwardgain)]

backwardrankingdict, backwardrankinglist = noderank.calc_simple_rank(
    backwardgain,
    backwardvariablelist,
    backwardbias,
    noderankdata,
    rank_method,
    package="simple",
)
