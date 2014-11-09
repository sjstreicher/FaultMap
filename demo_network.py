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

import networkgen

filesloc = json.load(open('config.json'))
saveloc = os.path.expanduser(filesloc['saveloc'])

weightgraph = nx.DiGraph()
gaingraph = nx.DiGraph()
onesgraph = nx.DiGraph()

#connections = np.matrix([[0, 0, 0, 1],
#                         [1, 0, 0, 1],
#                         [1, 0, 0, 0],
#                         [0, 1, 1, 0]])

#gainmatrix = np.matrix([[0.00, 0.00, 0.00, 0.35],
#                        [0.82, 0.00, 0.00, 0.63],
#                        [0.42, 0.00, 0.00, 0.00],
#                        [0.00, 0.00, 0.21, 0.00]])

#gainmatrix = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0],
#                        [0.0, 0.0, 0.0, 0.0, 0.0],
#                        [2.0, 1.0, 0.0, 0.0, 0.0],
#                        [0.0, 0.0, 1.0, 0.0, 0.0],
#                        [1.0, 2.0, 0.0, 0.0, 0.0]])

#gainmatrix = np.matrix([[1.0, 1.0, 1.0, 1.0, 1.0],
#                        [1.0, 1.0, 1.0, 1.0, 1.0],
#                        [1.0, 1.0, 1.0, 1.0, 1.0],
#                        [1.0, 1.0, 1.0, 1.0, 1.0],
#                        [1.0, 1.0, 1.0, 1.0, 1.0]])

dw = 10.

gainmatrix = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [dw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, dw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, dw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, dw, 0.0, 0.0, 0.0, 0.0, 0.0]])

variables = ['X 1', 'X 2', 'X 3', 'X 4', 'X 5', 'D 1', 'D 2', 'D 3', 'D 4']
#[_, gainmatrix, variables, _] = networkgen.series_equal_five()

n = gainmatrix.shape[0]

gainmatrix = np.asmatrix(gainmatrix, dtype=float)

#print gainmatrix

# Should the transpose happen before or after the column normalization?
# I have a feeling that it should definitely be before...


# Normalize the gainmatrix columns
for col in range(n):
    colsum = np.sum(abs(gainmatrix[:, col]))
    if colsum == 0:
        # Option :1 do nothing
        None
        # Option 2: equally connect to all other nodes
#        for row in range(n):
#            gainmatrix[row, col] = (1. / n)
    else:
        gainmatrix[:, col] = (gainmatrix[:, col]
                              / colsum)

print gainmatrix

#gainmatrix = gainmatrix[:-4, :-4]

#onesmatrix = np.ones_like(gainmatrix)

m = 0.99

relative_reset_vector = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
#relative_reset_vector = [1., 1., 1., 1., 1.]

relative_reset_vector_norm = np.asarray(relative_reset_vector, dtype=float) \
    / sum(relative_reset_vector)

#n = n-4

#resetmatrix = np.array([relative_reset_vector_norm, ]*n).T
resetmatrix = np.array([relative_reset_vector_norm, ]*(n))
#resetmatrix = (1./n) * np.ones_like(gainmatrix)
#print resetmatrix

weightmatrix = (m * gainmatrix) + ((1-m) * resetmatrix)

#print weightmatrix

weightmatrix = weightmatrix.T


# Normalize the weightmatrix columns
for col in range((n)):
    weightmatrix[:, col] = (weightmatrix[:, col]
                            / np.sum(abs(weightmatrix[:, col])))

#weightmatrix = weightmatrix[:-4, :-4]

#print weightmatrix


[eigval, eigvec] = np.linalg.eig(weightmatrix)
[eigval_gain, eigvec_gain] = np.linalg.eig(gainmatrix)
maxeigindex = np.argmax(eigval)

rankarray = eigvec[:, maxeigindex][:-4]

# Take absolute values of ranking values
rankarray = abs(np.asarray(rankarray))

#print rankarray
# This is the 1-dimensional array composed of rankings (normalised)
rankarray_norm = (1 / sum(rankarray)) * rankarray
# Remove the useless imaginary +0j
rankarray_norm = rankarray_norm.real

print rankarray_norm


#for col, colvar in enumerate(variables):
#    for row, rowvar in enumerate(variables):
#        # Create fully connected weighted graph for use with eigenvector
#        # centrality analysis
#        weightgraph.add_edge(rowvar, colvar,
#                             weight=weightmatrix[row, col])
#        onesgraph.add_edge(rowvar, colvar,
#                           weight=onesmatrix[row, col])
#        # Create sparsely connected graph based on significant edge weights
#        # only for use with Katz centrality analysis
#        if (gainmatrix[row, col] != 0.):
#            # The node order is source, sink according to
#            # the convention that columns are sources and rows are sinks
#            gaingraph.add_edge(rowvar, colvar, weight=gainmatrix[row, col])
#
#
#eig_rankingdict = nx.eigenvector_centrality(weightgraph)
#
#
#def norm_dict(dictionary):
#    dictionary_norm = dict()
#    entrysums = 0
#    for entry in dictionary:
#        entrysums += dictionary[entry]
#    for entry in dictionary:
#        dictionary_norm[entry] = dictionary[entry] / entrysums
#    return dictionary_norm
#
#
#eig_rankingdict_norm = norm_dict(eig_rankingdict)
#
#katz_rankingdict = nx.katz_centrality(gaingraph,
#                                      1.0, 1.0, 20000)
#
#katz_rankingdict_norm = norm_dict(katz_rankingdict)
#
#katz_rankingdict_weight = nx.katz_centrality(weightgraph,
#                                             0.99, 1.0, 20000)
#
##nx.write_gml(gaingraph, os.path.join(saveloc, "gaingraph.gml"))
##nx.write_gml(weightgraph, os.path.join(saveloc, "weightgraph.gml"))
##nx.draw(gaingraph)
##plt.show()
#
##nx.write_gml(weightgraph, os.path.join(saveloc, "weightgraph.gml"))
##nx.draw(weightgraph)
#
#plt.show()
#
#
#def calc_simple_rank(gainmatrix, variables, m, noderankdata):
#    """Constructs the ranking dictionary using the eigenvector approach
#    i.e. Ax = x where A is the local gain matrix.
#
#    Taking the absolute of the gainmatrix and normalizing to conform to
#    original LoopRank idea.
#
#    """
#
#    # Length of gain matrix = number of nodes
#    gainmatrix = np.abs(np.asarray(gainmatrix))
#    n = len(gainmatrix)
#    s_matrix = (1.0 / n) * np.ones((n, n))
#    # Basic PageRank algorithm
#    m_matrix = (1 - m) * gainmatrix + m * s_matrix
#    # Normalize the m-matrix columns
#    for col in range(n):
#        m_matrix[:, col] = m_matrix[:, col] / np.sum(abs(m_matrix[:, col]))
#    # Calculate eigenvalues and eigenvectors as usual
#    [eigval, eigvec] = np.linalg.eig(m_matrix)
#    maxeigindex = np.argmax(eigval)
#    # Store value for downstream checking
#    # TODO: Downstream checking not implemented yet
##    maxeig = eigval[maxeigindex].real
#    # Cuts array into the eigenvector corrosponding to the eigenvalue above
#    rankarray = eigvec[:, maxeigindex]
#    # Take absolute values of ranking values
#    rankarray = abs(rankarray)
#    # This is the 1-dimensional array composed of rankings (normalised)
#    rankarray = (1 / sum(rankarray)) * rankarray
#    # Remove the useless imaginary +0j
#    rankarray = rankarray.real
#
#    # Create a dictionary of the rankings with their respective nodes
#    # i.e. {NODE:RANKING}
#    rankingdict = dict(zip(variables, rankarray))
#
#    rankinglist = sorted(rankingdict.iteritems(), key=operator.itemgetter(1),
#                         reverse=True)
#
#    return rankingdict, rankinglist