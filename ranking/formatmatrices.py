"""This method is imported by looptest

@author: St. Elmo Wilken, Simon Streicher

"""

import numpy as np
import networkx as nx


def normalise_matrix(inputmatrix):
    """Normalises the absolute value of the input matrix in the columns
    such that all columns will sum to 1.


    """
    inputmatrix = abs(np.asarray(inputmatrix))  # Does not affect eigenvalues

    colsums = inputmatrix.sum(0)
    normalisedmatrix = inputmatrix/colsums
    # Remove nan values that entered due to division by zero
    normalisedmatrix[:, colsums == 0] = 0

    return normalisedmatrix


def removedummyvars(gainmatrix, connectionmatrix, variables,
                    dummy_var_no):
    """This method assumed the first variables up to dummy_var_no
    are the dummy variables.

    """
    nodummyvariablelist = []  # Necessary for a list copy
    nodummyvariablelist.extend(variables)
    nodummygain = gainmatrix.copy()
    nodummyconnection = connectionmatrix.copy()
    for index in range(dummy_var_no):
        nodummyvariablelist.pop(0)
        nodummygain = np.delete(nodummygain, 0, 0)
        nodummygain = np.delete(nodummygain, 0, 1)
        nodummyconnection = np.delete(nodummyconnection, 0, 0)
        nodummyconnection = np.delete(nodummyconnection, 0, 1)

    [r, c] = nodummyconnection.shape
    nodummy_nodes = r

    print "Number of dummy nodes: ", nodummy_nodes

    return nodummyvariablelist, nodummygain, nodummyconnection, nodummy_nodes


def addforwardscale(nodummyvariablelist, nodummygain, nodummyconnection,
                    nodummy_nodes):
    """This method adds a unit gain node to all nodes with an out-degree
    of 1; now all of these nodes should have an out-degree of 2.
    Therefore all nodes with pointers should have 2 or more edges pointing
    away from them.

    It uses the number of dummy variables to construct these gain,
    connection and variable name matrices.

    """
    m_graph = nx.DiGraph()
    # Construct the graph with connections
    for u in range(nodummy_nodes):
        for v in range(nodummy_nodes):
            if (nodummyconnection[u, v] != 0):
                m_graph.add_edge(nodummyvariablelist[v],
                                 nodummyvariablelist[u],
                                 weight=nodummygain[u, v])
    # Add connections where out degree == 1
    counter = 1
    for node in m_graph.nodes():
        if m_graph.out_degree(node) == 1:
            nameofscale = 'DV_forward' + str(counter)
            m_graph.add_edge(node, nameofscale, weight=1.0)
            counter += 1

    scaledforwardconnection = nx.to_numpy_matrix(m_graph, weight=None).T
    scaledforwardgain = nx.to_numpy_matrix(m_graph, weight='weight').T
    scaledforwardvariablelist = m_graph.nodes()

    return scaledforwardconnection, scaledforwardgain, \
        scaledforwardvariablelist


def addbackwardscale(nodummyvariablelist, nodummygain, nodummyconnection,
                     nodummy_nodes):
    """This method adds a unit gain node to all nodes with an out-degree
    of 1; now all of these nodes should have an out-degree of 2.
    Therefore all nodes with pointers should have 2 or more edges
    pointing away from them.

    It uses the number of dummy variables to construct these gain,
    connection and variable name matrices.

    This method transposes the original no dummy variables to
    generate the reverse option.

    """

    m_graph = nx.DiGraph()

    # Construct the graph with connections
    for u in range(nodummy_nodes):
        for v in range(nodummy_nodes):
            if (nodummyconnection.T[u, v] != 0):
                m_graph.add_edge(nodummyvariablelist[v],
                                 nodummyvariablelist[u],
                                 weight=nodummygain.T[u, v])

    # Now add connections where out degree == 1
    counter = 1
    for node in m_graph.nodes():
        if m_graph.out_degree(node) == 1:
            nameofscale = 'DV_backward' + str(counter)
            m_graph.add_edge(node, nameofscale, weight=1.0)
            counter += 1

    scaledbackwardconnection = nx.to_numpy_matrix(m_graph, weight=None).T
    scaledbackwardgain = nx.to_numpy_matrix(m_graph, weight='weight').T
    scaledbackwardvariablelist = m_graph.nodes()

    return scaledbackwardconnection, scaledbackwardgain, \
        scaledbackwardvariablelist







