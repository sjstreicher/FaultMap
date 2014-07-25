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

testgraph = nx.DiGraph()

variables = ['PV 1', 'PV 2', 'PV 3', 'PV 4']

#connections = np.matrix([[0, 0, 0, 1],
#                         [1, 0, 0, 1],
#                         [1, 0, 0, 0],
#                         [0, 0, 1, 0]])

gainmatrix = np.matrix([[0.00, 0.00, 0.00, 0.35],
                        [0.82, 0.00, 0.00, 0.63],
                        [0.42, 0.00, 0.00, 0.00],
                        [0.00, 0.00, 0.21, 0.00]])

gainmatrix = gainmatrix.T

m = 0.15
n = gainmatrix.shape[0]

resetmatrix = (1./n) * np.ones_like(gainmatrix)

weightmatrix = ((1-m) * gainmatrix) + (m * resetmatrix)

# Normalize the m-matrix columns
for col in range(n):
    weightmatrix[:, col] = weightmatrix[:, col] / np.sum(abs(weightmatrix[:, col]))

[eigval, eigvec] = np.linalg.eig(weightmatrix)
maxeigindex = np.argmax(eigval)

rankarray = eigvec[:, maxeigindex]

# Take absolute values of ranking values
rankarray = abs(rankarray)
# This is the 1-dimensional array composed of rankings (normalised)
rankarray_norm = (1 / sum(rankarray)) * rankarray
# Remove the useless imaginary +0j
rankarray_norm = rankarray_norm.real

for col, colvar in enumerate(variables):
    for row, rowvar in enumerate(variables):
        # The node order is source, sink according to
        # the convention that columns are sources and rows are sinks
        testgraph.add_edge(colvar, rowvar, weight=gainmatrix[row, col])

#nx.write_gml(testgraph, os.path.join(saveloc, "testgraph.gml"))
#nx.draw(testgraph)
#plt.show()

#rankingdict = nx.eigenvector_centrality(testgraph.reverse())

katz_rankingdict = nx.katz_centrality(testgraph.reverse())





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