"""Imported by controlranking

@author St. Elmo Wilken, Simon Streicher

"""

from numpy import ones, argmax, asarray
from numpy import linalg
import numpy as np
from operator import itemgetter
from itertools import izip
import networkx as nx
from matplotlib import pyplot as plt


def calc_simple_rank(gainmatrix, variables, m=0.15):
    """Constructs the ranking dictionary using the eigenvector approach
    i.e. Ax = x where A is the local gain matrix.
    
    m is the weight of the full connectivity matrix used to ensure
    graph is not sub-stochastic

    """
    # Length of gain matrix = number of nodes
    gainmatrix = asarray(gainmatrix)
    n = len(gainmatrix)
    s_matrix = (1.0 / n) * ones((n, n))
    # Basic PageRank algorithm
    m_matrix = (1 - m) * gainmatrix + m * s_matrix
    # Calculate eigenvalues and eigenvectors as usual
    [eigval, eigvec] = linalg.eig(m_matrix)
    maxeigindex = argmax(eigval)
    # Store value for downstream checking
    # TODO: Downstream checking not implemented yet
#    maxeig = eigval[maxeigindex].real
    # Cuts array into the eigenvector corrosponding to the eigenvalue above
    rankarray = eigvec[:, maxeigindex]
    # This is the 1-dimensional array composed of rankings (normalised)
    rankarray = (1 / sum(rankarray)) * rankarray
    # Remove the useless imaginary +0j
    rankarray = rankarray.real

    # Create a dictionary of the rankings with their respective nodes
    # i.e. {NODE:RANKING}
    rankingdict = dict(zip(variables, rankarray))
    
    rankinglist = sorted(rankingdict.iteritems(), key=itemgetter(1),
                         reverse=True)

    return rankingdict, rankinglist


def calc_blended_rank(forwardrank, backwardrank, variablelist,
                           alpha=0.35):
    """This method creates a blended ranking profile of the object."""
    rankingdict = dict()
    for variable in variablelist:
        rankingdict[variable] = abs(((1 - alpha) * forwardrank[variable] +
                                     (alpha) * backwardrank[variable]))

    totals = sum(rankingdict.values())
    # Normalise rankings
    for variable in variablelist:
        rankingdict[variable] = rankingdict[variable] / totals

    rankinglist = sorted(rankingdict.iteritems(), key=itemgetter(1),
                         reverse=True)
                         
    return rankingdict, rankinglist
    

def calc_transient_importancediffs(rankingdicts, variablelist):
    """Creates dictionary with a vector of successive differences in importance
    scores between boxes for each variable entry.

    """
    transientdict = dict()
    basevaldict = dict()
    for variable in variablelist:
        diffvect = np.empty((1, len(rankingdicts)-1))[0]
        diffvect[:] = np.NAN
        basevaldict[variable] = rankingdicts[0][variable]
        # Get initial previous importance
        prev_importance = basevaldict[variable]
        for index, rankingdict in enumerate(rankingdicts[1:]):
            diffvect[index] = rankingdict[variable] - prev_importance
            prev_importance = rankingdict[variable]
        transientdict[variable] = diffvect

    return transientdict, basevaldict


def plot_transient_importances(variables, transientdict, basevaldict):
    """Plots the transient importance for the specified variables.
    Plots both absolute rankings over time as well as ranking differences only.

    """
    transient_val_no = len(transientdict[variables[1]])
    # Transient rankings down in rows, each variable contained in a column
    diffplot = np.zeros((transient_val_no+1, len(variables)))
    absplot = np.zeros_like(diffplot)

    for index, variable in enumerate(variables):
        diffplot[:, index][1:] = transientdict[variable]
        absplot[0, index] = basevaldict[variable]
        absplot[:, index][1:] = diffplot[:, index][1:] + basevaldict[variable]

    bins = range(transient_val_no+1)

    plt.figure(1)
    plt.plot(bins, diffplot)
    plt.title('Relative importance variations over time')

    plt.figure(2)
    plt.plot(bins, absplot)
    plt.title('Absolute importance scores over time')

    return plt.figure(1), plt.figure(2)


def create_importance_graph(variablelist, closedconnections,
                            openconnections, gainmatrix, ranking):
    """Generates a graph containing the
    connectivity and importance of the system being displayed.
    Edge Attribute: color for control connection
    Node Attribute: node importance

    """

    opengraph = nx.DiGraph()

    for col, row in izip(openconnections.nonzero()[0],
                         openconnections.nonzero()[1]):
        opengraph.add_edge(variablelist[col], variablelist[row],
                           weight=gainmatrix[row, col])
    openedgelist = opengraph.edges()

    closedgraph = nx.DiGraph()
    for col, row in izip(closedconnections.nonzero()[0],
                         closedconnections.nonzero()[1]):
        newedge = (variablelist[col], variablelist[row])
        closedgraph.add_edge(*newedge, weight=gainmatrix[row, col],
                             controlloop=int(newedge not in openedgelist))

    for node in closedgraph.nodes():
        closedgraph.add_node(node, importance=ranking[node])

    return closedgraph, opengraph
