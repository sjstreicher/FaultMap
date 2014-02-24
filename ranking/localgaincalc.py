"""This method is imported by formatmatrices

@author: St. Elmo Wilken, Simon Streicher

"""

from numpy import array, transpose, zeros, hstack
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from random import random
from math import isnan


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
        fromfile = csv.reader(open(connection_loc))
        # Get rid of that first space. Now the variables are all stored.
        self.variables = fromfile.next()[1:]
        self.connectionmatrix = []
        for row in fromfile:
            # This gets rid of the variable name on each row
            # (its there to help create the matrix)
            col = row[1:]
            for element in col:
                if element == '1':
                    self.connectionmatrix.append(1)
                else:
                    self.connectionmatrix.append(0)

        self.n = len(self.variables)
        self.connectionmatrix = array(self.connectionmatrix).reshape(
            self.n, self.n)

    def calc_partialcor_gainmatrix(self, tags_tsdata):
        """This method strives to calculate the local gains in terms of the
        correlation between the variables. It uses the partial correlation
        method (Pearson's correlation).

        tags_tsdata contains the time series data for the tags with variables
        in colums and sampling instances in rows

        """
        fromfile = csv.reader(open(tags_tsdata), delimiter=',')

        # TODO: Use better iteration method
        dataholder = []
        for row in fromfile:
            rowfix = row
            for element in rowfix:
                dataholder.append(float(element))
        print "Number of variables: ", self.n
        print "Total number of data points: ", np.shape(dataholder)[0]

        self.inputdata = np.reshape(dataholder, (-1, self.n))
        # Used to use np.empty here, possible source of randomness
        self.correlationmatrix = array(np.zeros((self.n, self.n)))
        for i in range(self.n):
            for j in range(self.n):
                temp = pearsonr(self.inputdata[:, i], self.inputdata[:, j])[0]
                # TODO: This results in inconsistency and needs to be fixed
                if isnan(temp):
                    print "Unable to compute correlation between variables ", \
                    (i+1), " and ", (j+1)
                    break
                else:
                    self.correlationmatrix[i, j] = temp

        p_matrix = np.linalg.inv(self.correlationmatrix)
        self.partialcorrelationmatrix = array(np.zeros((self.n, self.n)))

        for i in range(self.n):
            for j in range(self.n):
                if self.connectionmatrix[i, j] == 1:
                    temp = (-1 * p_matrix[i, j] / (abs((p_matrix[i, i] *
                                                   p_matrix[j, j]))**0.5))
                    self.partialcorrelationmatrix[i, j] = temp
                else:
                    self.partialcorrelationmatrix[i, j] = 0