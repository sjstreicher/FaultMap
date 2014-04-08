"""This method is imported by looptest

@author: St. Elmo Wilken, Simon Streicher

"""

import numpy as np
import networkx as nx
import h5py
import csv


def read_connectionmatrix(connection_loc):
    """This method imports the connection scheme for the data.
    The format should be:
    empty space, var1, var2, etc... (first row)
    var1, value, value, value, etc... (second row)
    var2, value, value, value, etc... (third row)
    etc...

    Value is 1 if column variable points to row variable
    (causal relationship)
    Value is 0 otherwise

    This method also stores the names of all the variables in the
    connection matrix.
    It is important that the order of the variables in the
    connection matrix match those in the data matrix.

    """
    with open(connection_loc) as f:
        variables = csv.reader(f).next()[1:]
        connectionmatrix = np.genfromtxt(f, delimiter=',')[:, 1:]

    return variables, connectionmatrix


def read_gainmatrix(gainmatrix_loc):
    """This method a gainmatrix scheme for a specific scenario.


    """
    with open(gainmatrix_loc) as f:
        gainmatrix = np.genfromtxt(f, delimiter=',')

    return gainmatrix


def buildcase(dummyweight, digraph, name, dummycreation):

    if dummycreation:
        counter = 1
        for node in digraph.nodes():
            if digraph.out_degree(node) == 1:
                # TODO: Investigate the effect of different weights

                nameofscale = name + str(counter)
                digraph.add_edge(node, nameofscale, weight=dummyweight)
                counter += 1

    connection = nx.to_numpy_matrix(digraph, weight=None).T
    gain = nx.to_numpy_matrix(digraph, weight='weight').T
    variablelist = digraph.nodes()
    return np.array(connection), gain, variablelist


def buildgraph(variables, gainmatrix, connections):
    digraph = nx.DiGraph()
    # Construct the graph with connections
    for col, colvar in enumerate(variables):
        for row, rowvar in enumerate(variables):
            # The node order is source, sink according to
            # the convention that columns are sources and rows are sinks
            if (connections[row, col] != 0):
                digraph.add_edge(colvar, rowvar, weight=gainmatrix[row, col])
    return digraph


def rankforward(variables, gainmatrix, connections,
                dummyweight, dummycreation):
    """This method adds a unit gain node to all nodes with an out-degree
    of 1; now all of these nodes should have an out-degree of 2.
    Therefore all nodes with pointers should have 2 or more edges pointing
    away from them.

    It uses the number of dummy variables to construct these gain,
    connection and variable name matrices.
    """

    #TODO: Rework calls of this code to reduce redundancy

    digraph = buildgraph(variables, gainmatrix, connections)
    return buildcase(dummyweight, digraph, 'DV FWD ', dummycreation)


def rankbackward(variables, gainmatrix, connections,
                 dummyweight, dummycreation):
    """This method adds a unit gain node to all nodes with an out-degree
    of 1; now all of these nodes should have an out-degree of 2.
    Therefore all nodes with pointers should have 2 or more edges
    pointing away from them.

    It uses the number of dummy variables to construct these gain,
    connection and variable name matrices.

    This method transposes the original no dummy variables to
    generate the reverse option.

    """

    #TODO: Rework calls of this code to reduce redundancy

    digraph = buildgraph(variables, gainmatrix.T, connections.T)
    return buildcase(dummyweight, digraph, 'DV BWD ', dummycreation)


def split_tsdata(tags_tsdata, datasetname, samplerate, boxsize, boxnum):
    """Splits the tags_tsdata into arrays useful for analysing the change of
    weights over time.

    samplerate is the rate of sampling in time units
    boxsize is the size of each returned dataset in time units
    boxnum is the number of boxes that need to be analyzed

    Boxes is evenly distributed over the provided dataset.
    The boxes will overlap if boxsize*boxnum is more than the simulated time,
    and will have spaced between them if it is less.


    """
    # Import the data as a numpy array
    inputdata = np.array(h5py.File(tags_tsdata, 'r')[datasetname])
    # Get total number of samples
    samples = len(inputdata)
#    print "Number of samples: ", samples
    # Convert boxsize to number of samples
    boxsizesamples = int(round(boxsize / samplerate))
#    print "Box size in samples: ", boxsizesamples
    # Calculate starting index for each box
    boxstartindex = np.empty((1, boxnum))[0]
    boxstartindex[:] = np.NAN
    boxstartindex[0] = 0
    boxstartindex[-1] = samples - boxsizesamples
    samplesbetween = int(round(boxstartindex[-1]/(boxnum-1)))
    boxstartindex[1:-1] = [(samplesbetween * index)
                           for index in range(1, boxnum-1)]
    boxes = [inputdata[int(boxstartindex[i]):int(boxstartindex[i]) + int(boxsizesamples)]
             for i in range(int(boxnum))]
    return boxes