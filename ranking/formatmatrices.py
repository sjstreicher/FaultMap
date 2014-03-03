"""This method is imported by looptest

@author: St. Elmo Wilken, Simon Streicher

"""

from localgaincalc import LocalGains
import numpy as np
import networkx as nx


class FormatMatrix:
    """This class formats the input connection, gain and variable
    name matrices.

    For self made test systems you will have dummy_var_no
    not equal to 0; this is especially true if you have recycle streams.

    This class will also "scale" the nodes of out-vertex == 1 nodes.

    """

    def __init__(self, connection_loc, tags_tsdata,
                 dummy_var_no):
        """This class assumes you input a connection matrix (connection_loc)
        ordered according to the tags_tsdata matrix with the
        dummy_var_no variables being the first number of variables which are
        dummy variables to be stripped.

        connection_loc is an adjacency matrix indicating the presence of
        connections in the system

        tags_tsdata contains the time series data for the tags with variables
        in colums and sampling instances in rows

        """
        self.init_system(connection_loc, tags_tsdata)
        self.removedummyvars(dummy_var_no)
        self.addforwardscale()
        self.addbackwardscale()

    def init_system(self, connection_loc, tags_tsdata):
        """This method creates the orignal gain matrix (including dummy gains)
        and the original connection matrix.

        """
        original = LocalGains(connection_loc, tags_tsdata)
        self.originalcorrelation = original.correlationmatrix
        self.originalgain = original.partialcorrelationmatrix
        self.variablelist = original.variables
        self.originalconnection = original.connectionmatrix

    def removedummyvars(self, dummy_var_no):
        """This method assumed the first variables up to dummy_var_no
        are the dummy variables.

        """
        self.nodummyvariablelist = []  # Necessary for a list copy
        self.nodummyvariablelist.extend(self.variablelist)
        self.nodummygain = self.originalgain.copy()
        self.nodummyconnection = self.originalconnection.copy()
        for index in range(dummy_var_no):
            self.nodummyvariablelist.pop(0)
            self.nodummygain = np.delete(self.nodummygain, 0, 0)
            self.nodummygain = np.delete(self.nodummygain, 0, 1)
            self.nodummyconnection = np.delete(self.nodummyconnection, 0, 0)
            self.nodummyconnection = np.delete(self.nodummyconnection, 0, 1)

        [r, c] = self.nodummyconnection.shape
        self.nodummy_nodes = r

    def addforwardscale(self):
        """This method adds a unit gain node to all nodes with an out-degree
        of 1; now all of these nodes should have an out-degree of 2.
        Therefore all nodes with pointers should have 2 or more edges pointing
        away from them.

        It uses the number of dummy variables to construct these gain,
        connection and variable name matrices.

        """
        m_graph = nx.DiGraph()
        # Construct the graph with connections
        for u in range(self.nodummy_nodes):
            for v in range(self.nodummy_nodes):
                if (self.nodummyconnection[u, v] != 0):
                    m_graph.add_edge(self.nodummyvariablelist[v],
                                     self.nodummyvariablelist[u],
                                     weight=self.nodummygain[u, v])
        # Add connections where out degree == 1
        counter = 1
        for node in m_graph.nodes():
            if m_graph.out_degree(node) == 1:
                nameofscale = 'DV' + str(counter)
                m_graph.add_edge(node, nameofscale, weight=1.0)
                counter = counter + 1

        self.scaledforwardconnection = nx.to_numpy_matrix(m_graph, weight=None).T
        self.scaledforwardgain = nx.to_numpy_matrix(m_graph, weight='weight').T
        self.scaledforwardvariablelist = m_graph.nodes()

    def addbackwardscale(self):
        """This method adds a unit gain node to all nodes with an out-degree
        of 1; now all of these nodes should have an out-degree of 2.
        Therefore all nodes with pointers should have 2 or more edges
        pointing away from them.

        It uses the number of dummy variables to construct these gain,
        connection and variable name matrices.

        Additionally, this method transposes the original no dummy variables to
        generate the reverse option.

        """

        m_graph = nx.DiGraph()
        transposedconnection = self.nodummyconnection.T
        transposedgain = self.nodummygain.T

        # Construct the graph with connections
        for u in range(self.nodummy_nodes):
            for v in range(self.nodummy_nodes):
                if (transposedconnection[u, v] != 0):
                    m_graph.add_edge(self.nodummyvariablelist[v],
                                     self.nodummyvariablelist[u],
                                     weight=transposedgain[u, v])

        # Now add connections where out degree == 1
        counter = 1
        for node in m_graph.nodes():
            if m_graph.out_degree(node) == 1:
                nameofscale = 'DV' + str(counter)
                m_graph.add_edge(node, nameofscale, weight=1.0)
                counter = counter + 1

        self.scaledbackwardconnection = nx.to_numpy_matrix(m_graph, weight=None).T
        self.scaledbackwardgain = nx.to_numpy_matrix(m_graph, weight='weight').T
        self.scaledbackwardvariablelist = m_graph.nodes()