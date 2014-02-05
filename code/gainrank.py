# -*- coding: utf-8 -*-
"""
@author St. Elmo Wilken, Simon Streicher
"""

from numpy import ones, argmax
from numpy import linalg


class GainRanking:
    """Calculates the importance of variables based on their first order
    local gains.

    """
    def __init__(self, localgainmatrix, listofvariables):
        """Creates a rankings dictionary with the variables as keys and the
        normlised rank as value.

        """
        self.gmatrix = localgainmatrix
        self.gvariables = listofvariables
        self.construct_rank_array()

    def construct_rank_array(self):
        """Constructs the ranking dictionary using the eigenvector approach
        i.e. Ax = x where A is the local gain matrix.

        """
        # Length of gain matrix = number of nodes
        self.n = len(self.gmatrix)
        S = (1.0 / self.n) * ones((self.n, self.n))
        m = 0.15
        # Basic PageRank algorithm
        self.M = (1 - m) * self.gmatrix + m * S
        # Calculate eigenvalues and eigenvectors as usual
        [eigval, eigvec] = linalg.eig(self.M)

        maxeigindex = argmax(eigval)
        # Store value for downstream checking
        self.maxeig = eigval[maxeigindex].real
        # Cuts array into the eigenvector corrosponding to the eigenvalue above
        self.rankarray = eigvec[:, maxeigindex]
        # This is the 1-dimensional array composed of rankings (normalised)
        self.rankarray = (1 / sum(self.rankarray)) * self.rankarray
        # Remove the useless imaginary +0j
        self.rankarray = self.rankarray.real

        # Create a dictionary of the rankings with their respective nodes
        # i.e. {NODE:RANKING}
        self.rankdict = dict(zip(self.gvariables, self.rankarray))

        # TODO: Include a test here
        # print(self.rankdict)