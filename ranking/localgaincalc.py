"""This method is imported by formatmatrices

@author: St. Elmo Wilken, Simon Streicher

"""

import csv
import numpy as np

class LocalGains:
    """This class:
        1) Imports the connection matrix from file
           (adjacency matrix indicating relevant connections)
        2) Imports the tags time series data matrix from file
        3) Constructs the linear local gain matrix
           (only partial correlation is supported at present)

    """

    def __init__(self, connection_loc, tags_tsdata):
        """This constructor creates matrices in memory of the connection and
        local gain inputs.

        """
        self.create_connectionmatrix(connection_loc)
        self.calc_partialcor_gainmatrix(tags_tsdata)

    def create_connectionmatrix(self, connection_loc):
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
            self.variables = csv.reader(f).next()[1:]
            self.connectionmatrix = np.genfromtxt(f, delimiter=',')[:, 1:]
            self.n = len(self.variables)

    def calc_partialcor_gainmatrix(self, tags_tsdata):
        """This method strives to calculate the local gains in terms of the
        correlation between the variables. It uses the partial correlation
        method (Pearson's correlation).

        tags_tsdata contains the time series data for the tags with variables
        in colums and sampling instances in rows

        """
        inputdata = np.loadtxt(tags_tsdata, delimiter=',')
        print "Number of variables: ", self.n
        print "Total number of data points: ", inputdata.size

        self.correlationmatrix = np.corrcoef(inputdata.T)

        np.savetxt("correlation.csv", self.correlationmatrix, delimiter=',')
        p_matrix = np.linalg.inv(self.correlationmatrix)
        d = p_matrix.diagonal()
        self.partialcorrelationmatrix = np.where(self.connectionmatrix, 
                                                 -p_matrix/np.abs(np.sqrt(np.outer(d, d))),
                                                 0)

        np.savetxt("partialcorr.csv", self.partialcorrelationmatrix, delimiter=',')