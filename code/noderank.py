"""This was later contained in other modules.

@author: St. Elmo Wilken, Simon Streicher
"""
from numpy import array, zeros


class NodeRanking:
    """Ranks nodes given a gain matrix, list of variables,
    a list with relative instrinsic values and a value for alpha.

    """

    def __init__(self, gainmat, varmat, intvalmat, alpha):
        # Get normalised gain matrix n*n array
        self.g_matrix = array(gainmat)
        # Get ordered variables wrt g_matrix 1*n array
        self.g_vars = varmat
        # Get the weighting of the intrinsic value analysis relative to
        # the gain analysis
        self.alpha = alpha
        # Get the intrinsic value of the variables 1*n array
        self.intrinsicvalue = intvalmat
        # Methods
        self.construct_adjacency_array()
        self.construct_intrinsicvalue_array()
        self.construct_rankarray()

    def construct_adjacency_array(self):
        """Constructs the adjacency array"""
        # Number of rows of gain matrix = number of nodes
        self.size = len(self.g_matrix)
        self.adjacency_array = array(zeros((self.size, self.size)))
        for i in range(self.size):
            for j in range(self.size):
                if self.g_matrix[i, j] != 0:
                    self.adjacency_array[i, j] = 1

    def construct_intrinsicvalue_array(self):
        """Constructs and normalise the intrinsic value array.
        The intvalmat input gives information on which variables
        are more important.
        Basically a list of factors.

        """
        self.intrinsicvalue_array = array(zeros((self.size, self.size)))
        for i in range(self.size):
            for j in range(self.size):
                self.intrinsicvalue_array[i, j] = \
                    self.adjacency_array[i, j]*self.intrinsicvalue[i]
        # Normalise the result of above
        self.totals = [sum(self.intrinsicvalue_array[:, j])
                       for j in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                self.intrinsicvalue_array[i, j] = \
                    self.intrinsicvalue_array[i, j]*(1.0/self.totals[j])

    def construct_rankarray(self):
        """Constructs the rank array"""
        from numpy import ones, argmax
        from numpy import linalg
        # M = (1 - m)*A + m*S
        # A = (1 - alpha)*B + alpha*C
        # B = normalised gain matrix
        # C = normalised intrinsic value matrix
        # S = gets rid of sub stochasticity for rows of all 0
        self.a_matrix = (1 - self.alpha)*self.g_matrix + \
            self.alpha*self.intrinsicvalue_array
        self.s_matrix = (1.0/self.size)*ones((self.size, self.size))
        mval = 0.15
        # Basic PageRank algorithm
        self.m_matrix = (1 - mval)*self.a_matrix + mval*self.s_matrix
        # Calculate eigenvalues, eigenvectors as usual
        [eigval, eigvec] = linalg.eig(self.m_matrix)

        maxeigindex = argmax(eigval)
        # Store value for downstream checking
        self.maxeig = eigval[maxeigindex].real
        # Cuts array into the eigenvector corrosponding to the eigenvalue above
        self.rank_array = eigvec[:, maxeigindex]
        # This is the 1-dimensional array composed of rankings (normalised)
        self.rank_array = (1/sum(self.rank_array))*self.rank_array
        # Remove the useless imaginary +0j
        self.rank_array = self.rank_array.real

if __name__ == "__main__":

    TEST_GAINMAT = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0],
                    [1.0/3, 0, 1.0/3, 0, 0, 0, 1.0/3, 0, 0, 0, 0, 0, 0, 1.0/3],
                    [1.0/3, 0, 1.0/3, 0, 1, 0, 1.0/3, 0, 0, 0, 0, 0, 0, 1.0/3],
                    [0, 1, 0, 1, 0, 0.5, 0, 0, 0, 0, 0, 0, 1, 0],
                    [1.0/3, 0, 1.0/3, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 1.0/3],
                    [0, 0, 0, 0, 0, 0, 1.0/3, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0]]
    TEST_VARMAT = ["T1", "F1", "T2", "F2", "R1", "X1", "F3", "T3", "F4", "T4",
                   "F6", "T6", "F5", "T5"]
    # All the temperature variable are 4 times more important
    # than the other ones (for safety reasons)
    TEST_INTVALMAT = [4, 1, 4, 1, 1, 1, 1, 4, 1, 4, 1, 4, 1, 4]
    TEST = NodeRanking(TEST_GAINMAT, TEST_VARMAT, TEST_INTVALMAT, 0.5)
    print(TEST.rank_array)