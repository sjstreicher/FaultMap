"""This method is imported by formatmatrices

@author: St. Elmo Wilken, Simon Streicher

"""

import csv
import numpy as np
import h5py


def create_connectionmatrix(connection_loc):
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


def calc_partialcor_gainmatrix(connectionmatrix, tags_tsdata, datasetname):
    """Calculates the local gains in terms of the partial (Pearson's)
    correlation between the variables.

    connectionmatrix is the adjacency matrix

    tags_tsdata contains the time series data for the tags with variables
    in colums and sampling instances in rows

    """
    
    inputdata = np.array(h5py.File(tags_tsdata, 'r')[datasetname])
#    inputdata = np.loadtxt(tags_tsdata, delimiter=',')
    print "Total number of data points: ", inputdata.size
    # Calculate correlation matrix
    correlationmatrix = np.corrcoef(inputdata.T)
    # Calculate partial correlation matrix
    p_matrix = np.linalg.inv(correlationmatrix)
    d = p_matrix.diagonal()
    partialcorrelationmatrix = \
        np.where(connectionmatrix, -p_matrix/np.abs(np.sqrt(np.outer(d, d))),
                 0)
    # Removing the sign generally seems to destroy information
    # However, there is a concern that it can be influenced by time delays
    # TODO: Search for the maximum absolute value over a range of time delays
    # and then select the maximum with correct sign.
#    partialcorrelationmatrix = abs(partialcorrelationmatrix)

    return correlationmatrix, partialcorrelationmatrix


def calc_transentropy_gainmatrix(connectionmatrix, tags_tsdata):
    """Calculates the local gains in terms of the transfer entropy between
    the variables.

    connectionmatrix is the adjacency matrix

    tags_tsdata contains the time series data for the tags with variables
    in colums and sampling instances in rows

    """

    #TODO: To be completed

    return None
