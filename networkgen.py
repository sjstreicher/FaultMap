"""
Created on Mon Feb 24 15:27:21 2014

@author: Simon Streicher
"""

from ranking.formatmatrices import buildgraph
import networkx as nx
import numpy as np
import json
import os
from config_setup import ensure_existance

#filesloc = json.load(open('config.json'))
#saveloc = os.path.expanduser(filesloc['saveloc'])


def fullconn_equal():
    """Creates a fully connected 5x5 digraph with unit weights
    on all edges.

    """

    variables = ['X 1', 'X 2', 'X 3', 'X 4', 'X 5']

    connections = np.ones((5, 5))
    gainmatrix = np.ones((5, 5))

    testgraph = buildgraph(variables, gainmatrix, connections)

#    nx.write_gml(testgraph, os.path.join(saveloc, 'testgraphs',
#                                         "fullconn_equal.gml"))
#    nx.draw(testgraph)

    return connections, gainmatrix, variables, testgraph


def series_equal():
    """Creates five tags connected in series with unit weights on
    all edges.

    """

    variables = ['X 1', 'X 2', 'X 3', 'X 4', 'X 5']

    connections = np.array([[0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0]])

    gainmatrix = np.array([[0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0]])

    testgraph = buildgraph(variables, gainmatrix, connections)

#    nx.write_gml(testgraph, os.path.join(saveloc, 'testgraphs',
#                                         "series_equal.gml"))
#    nx.draw(testgraph)

    return connections, gainmatrix, variables, testgraph


def series_incomingon2nd():
    """Creates five tags connected in series with unit weights on
    all edges.

    """

    variables = ['X 1', 'X 2', 'X 3', 'X 4', 'X 5', 'I 1']

    connections = np.array([[0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 1],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0]])

    gainmatrix = np.array([[0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 1],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0]])

    testgraph = buildgraph(variables, gainmatrix, connections)

#    nx.write_gml(testgraph, os.path.join(saveloc, 'testgraphs',
#                                         "series_incomingon2nd.gml"))
#    nx.draw(testgraph)

    return connections, gainmatrix, variables, testgraph


def series_incomingon3rd():
    """Creates five tags connected in series with unit weights on
    all edges.

    """

    variables = ['X 1', 'X 2', 'X 3', 'X 4', 'X 5', 'I 1']

    connections = np.array([[0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 1],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0]])

    gainmatrix = np.array([[0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 1],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0]])

    testgraph = buildgraph(variables, gainmatrix, connections)

#    nx.write_gml(testgraph, os.path.join(saveloc, 'testgraphs',
#                                         "series_incomingon3rd.gml"))
#    nx.draw(testgraph)

    return connections, gainmatrix, variables, testgraph


def series_incomingon2ndand3rd():
    """Creates five tags connected in series with unit weights on
    all edges.

    """

    variables = ['X 1', 'X 2', 'X 3', 'X 4', 'X 5', 'I 1']

    connections = np.array([[0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 1],
                            [0, 1, 0, 0, 0, 1],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0]])

    gainmatrix = np.array([[0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 1],
                           [0, 1, 0, 0, 0, 1],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0]])

    testgraph = buildgraph(variables, gainmatrix, connections)

#    nx.write_gml(testgraph, os.path.join(saveloc, 'testgraphs',
#                                         "series_incomingon2ndand3rd.gml"))
#    nx.draw(testgraph)

    return connections, gainmatrix, variables, testgraph


def series_incomingon2ndand4th():
    """Creates five tags connected in series with unit weights on
    all edges.

    """

    variables = ['X 1', 'X 2', 'X 3', 'X 4', 'X 5', 'I 1']

    connections = np.array([[0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 1],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0]])

    gainmatrix = np.array([[0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 1],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 1],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0]])

    testgraph = buildgraph(variables, gainmatrix, connections)

#    nx.write_gml(testgraph, os.path.join(saveloc, 'testgraphs',
#                                         "series_incomingon2ndand4th.gml"))
#    nx.draw(testgraph)

    return connections, gainmatrix, variables, testgraph


def series_disjoint_equal():
    """Creates two sets of three tags connected in series with unit weights on
    all edges.

    """

    variables = ['X 1', 'X 2', 'X 3', 'Y 1', 'Y 2', 'Y 3']

    connections = np.array([[0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0]])

    gainmatrix = np.array([[0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0]])

    testgraph = buildgraph(variables, gainmatrix, connections)

#    nx.write_gml(testgraph, os.path.join(saveloc, 'testgraphs',
#                                         "series_disjoint_equal.gml"))
#    nx.draw(testgraph)

    return connections, gainmatrix, variables, testgraph


def series_disjoint_unequal():
    """Creates two sets of three tags connected in series with unit weights on
    the edges of one series and weights of 2 on that of the other.

    """

    variables = ['X 1', 'X 2', 'X 3', 'Y 1', 'Y 2', 'Y 3']

    connections = np.array([[0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0]])

    gainmatrix = np.array([[0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 2, 0, 0],
                           [0, 0, 0, 0, 2, 0]])

    testgraph = buildgraph(variables, gainmatrix, connections)

#    nx.write_gml(testgraph, os.path.join(saveloc, 'testgraphs',
#                                         "series_disjoint_equal.gml"))
#    nx.draw(testgraph)

    return connections, gainmatrix, variables, testgraph


def series_disjoint_unequalsource():
    """Creates two sets of three tags connected in series with unit weights on
    the edges of one series and weights of 2 on that of the other.

    """

    variables = ['X 1', 'X 2', 'X 3', 'Y 1', 'Y 2', 'Y 3', 'I 1']

    connections = np.array([[0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]])

    gainmatrix = np.array([[0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 2],
                           [0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]])

    testgraph = buildgraph(variables, gainmatrix, connections)

#    nx.write_gml(testgraph, os.path.join(saveloc, 'testgraphs',
#                                         "series_disjoint_equal.gml"))
#    nx.draw(testgraph)

    return connections, gainmatrix, variables, testgraph


def circle_equal():
    """Creates five tags connected in a circle with unit weights on
    all edges.

    """

    variables = ['X 1', 'X 2', 'X 3', 'X 4', 'X 5']

    connections = np.array([[0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0]])

    gainmatrix = np.array([[0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0]])

    testgraph = buildgraph(variables, gainmatrix, connections)

#    savedir = ensure_existance(os.path.join(saveloc, 'testgraphs'), make=True)
#    nx.write_gml(testgraph, os.path.join(savedir, "circle_equal.gml"))
#    nx.draw(testgraph)

    return connections, gainmatrix, variables, testgraph


def circle_unequalon2to3():
    """Creates five tags connected in a circle with unit weights on
    all edges except for one.

    """

    variables = ['X 1', 'X 2', 'X 3', 'X 4', 'X 5']

    connections = np.array([[0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0]])

    gainmatrix = np.array([[0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0],
                           [0, 2, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0]])

    testgraph = buildgraph(variables, gainmatrix, connections)

#    nx.write_gml(testgraph, os.path.join(saveloc, 'testgraphs',
#                                         "circle_unequalon2to3.gml"))
#    nx.draw(testgraph)

    return connections, gainmatrix, variables, testgraph
