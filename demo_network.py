# -*- coding: utf-8 -*-
"""
Created on Thu Mar 06 14:18:54 2014

@author: Simon
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
logging.basicConfig(level=logging.INFO)

import os

filesloc = json.load(open('config.json'))
saveloc = os.path.expanduser(filesloc['saveloc'])

weightgraph = nx.DiGraph()
gaingraph = nx.DiGraph()
onesgraph = nx.DiGraph()

variables = ['PV 1', 'PV 2', 'PV 3', 'PV 4']

connections = np.matrix([[0, 0, 0, 1],
                         [1, 0, 0, 1],
                         [1, 0, 0, 0],
                         [0, 1, 1, 0]])


gainmatrix = np.matrix([[0.00, 0.00, 0.00, 0.35],
                        [0.82, 0.00, 0.00, 0.63],
                        [0.42, 0.00, 0.00, 0.00],
                        [0.00, 0.00, 0.21, 0.00]])

# Should the transpose happen before or after the column normalization?
# I have a feeling that it should definitely be before...

gainmatrix = gainmatrix.T

n = gainmatrix.shape[0]

# Normalize the gainmatrix columns
#for col in range(n):
#    colsum = np.sum(abs(gainmatrix[:, col]))
#    if colsum == 0:
#        # Option 1 do nothing
#        None
#        # Option two: equally connect to all other nodes
##        gainmatrix[:, col] = (np.ones([n, 1]) / n)
#    else:
#        gainmatrix[:, col] = (gainmatrix[:, col]
#                              / colsum)

onesmatrix = np.ones_like(gainmatrix)

m = 0.15

resetmatrix = (1./n) * np.ones_like(gainmatrix)

weightmatrix = ((1-m) * gainmatrix) + (m * resetmatrix)

# Normalize the m-matrix columns
for col in range(n):
    weightmatrix[:, col] = (weightmatrix[:, col]
                            / np.sum(abs(weightmatrix[:, col])))

#weightmatrix = weightmatrix.T

[eigval, eigvec] = np.linalg.eig(weightmatrix)
maxeigindex = np.argmax(eigval)

rankarray = eigvec[:, maxeigindex]

# Take absolute values of ranking values
rankarray = abs(np.asarray(rankarray))
# This is the 1-dimensional array composed of rankings (normalised)
rankarray_norm = (1 / sum(rankarray)) * rankarray
# Remove the useless imaginary +0j
rankarray_norm = rankarray_norm.real

for col, colvar in enumerate(variables):
    for row, rowvar in enumerate(variables):
        # Create fully connected weighted graph for use with eigenvector
        # centrality analysis
        weightgraph.add_edge(rowvar, colvar,
                             weight=weightmatrix[row, col])
        onesgraph.add_edge(rowvar, colvar,
                           weight=onesmatrix[row, col])
        # Create sparsely connected graph based on significant edge weights
        # only for use with Katz centrality analysis
        if (gainmatrix[row, col] != 0.):
            # The node order is source, sink according to
            # the convention that columns are sources and rows are sinks
            gaingraph.add_edge(rowvar, colvar, weight=gainmatrix[row, col])


eig_rankingdict = nx.eigenvector_centrality(weightgraph)


katz_rankingdict = nx.katz_centrality(gaingraph, 0.1, 1.0, 20000)

katz_rankingdict_weight = nx.katz_centrality(weightgraph, 0.9, 1.0, 20000)

#nx.write_gml(gaingraph, os.path.join(saveloc, "gaingraph.gml"))
#nx.draw(gaingraph)
#plt.show()

#nx.write_gml(weightgraph, os.path.join(saveloc, "weightgraph.gml"))
#nx.draw(weightgraph)

plt.show()


def calc_simple_rank(gainmatrix, variables, m, noderankdata):
    """Constructs the ranking dictionary using the eigenvector approach
    i.e. Ax = x where A is the local gain matrix.

    Taking the absolute of the gainmatrix and normalizing to conform to
    original LoopRank idea.

    """

    # Length of gain matrix = number of nodes
    gainmatrix = np.abs(np.asarray(gainmatrix))
    n = len(gainmatrix)
    s_matrix = (1.0 / n) * np.ones((n, n))
    # Basic PageRank algorithm
    m_matrix = (1 - m) * gainmatrix + m * s_matrix
    # Normalize the m-matrix columns
    for col in range(n):
        m_matrix[:, col] = m_matrix[:, col] / np.sum(abs(m_matrix[:, col]))
    # Calculate eigenvalues and eigenvectors as usual
    [eigval, eigvec] = np.linalg.eig(m_matrix)
    maxeigindex = np.argmax(eigval)
    # Store value for downstream checking
    # TODO: Downstream checking not implemented yet
#    maxeig = eigval[maxeigindex].real
    # Cuts array into the eigenvector corrosponding to the eigenvalue above
    rankarray = eigvec[:, maxeigindex]
    # Take absolute values of ranking values
    rankarray = abs(rankarray)
    # This is the 1-dimensional array composed of rankings (normalised)
    rankarray = (1 / sum(rankarray)) * rankarray
    # Remove the useless imaginary +0j
    rankarray = rankarray.real

    # Create a dictionary of the rankings with their respective nodes
    # i.e. {NODE:RANKING}
    rankingdict = dict(zip(variables, rankarray))

    rankinglist = sorted(rankingdict.iteritems(), key=operator.itemgetter(1),
                         reverse=True)

    return rankingdict, rankinglist