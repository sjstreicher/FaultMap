"""This method is imported by looptest

@author: St. Elmo Wilken, Simon Streicher

"""

#from visualise import visualiseOpenLoopSystem
from gainrank import GainRanking
from numpy import array
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
import csv


class LoopRanking:
    """This class:
    1) Ranks the importance of nodes in a system with control
    1.a) Use the local gains to determine importance
    1.b) Use partial correlation information to determine importance
    2) Determine the change of importance when variables change

    """

    # TODO: Generalize alpha definition
    def __init__(self, fgainmatrix, fvariablenames, fconnectionmatrix,
                 bgainmatrix, bvariablenames, bconnectionmatrix,
                 nodummyvariablelist, alpha=0.35):
        """This constructor creates grapgs with associated node importances
        based on:
        1) Local gain information
        2) Partial correlation data

        """

        self.forwardgain = GainRanking(self.normalise_matrix(fgainmatrix),
                                       fvariablenames)
        self.backwardgain = GainRanking(self.normalise_matrix(bgainmatrix),
                                        bvariablenames)
        self.create_blended_ranking(nodummyvariablelist, alpha)

    def create_blended_ranking(self, nodummyvariablelist, alpha=0.35):
        """This method creates a blended ranking profile of the object."""
        self.variablelist = nodummyvariablelist
        self.blendedranking = dict()
        for variable in nodummyvariablelist:
            self.blendedranking[variable] = ((1 - alpha) *
                 self.forwardgain.rankdict[variable] + (alpha) *
                 self.backwardgain.rankdict[variable])

        slist = sorted(self.blendedranking.iteritems(), key=itemgetter(1),
                       reverse=True)
        writer = csv.writer(open('importances.csv', 'wb'))
        for x in slist:
            writer.writerow(x)
            print(x)

        print("Done with controlled importances")

    def normalise_matrix(self, inputmatrix):
        """Normalises the absolute value of the input matrix in the columns
        such that all columns will sum to 1.

        """
        [r, c] = inputmatrix.shape
        inputmatrix = abs(inputmatrix)  # Does not affect eigenvalues
        normalisedmatrix = []

        for col in range(c):
            colsum = float(sum(inputmatrix[:, col]))
            for row in range(r):
                if (colsum != 0):
                    normalisedmatrix.append(inputmatrix[row, col] / colsum)
                else:
                    normalisedmatrix.append(0.0)
        normalisedmatrix = array(normalisedmatrix).reshape(r, c).T
        return normalisedmatrix

    def display_control_importances(self, nocontrolconnectionmatrix,
                                    controlconnectionmatrix):
        """Generates a graph containing the
        connectivity and importance of the system being displayed.
        Edge Attribute: color for control connection
        Node Attribute: node importance

        """
        nc_graph = nx.DiGraph()
        n = len(self.variablelist)
        for u in range(n):
            for v in range(n):
                if nocontrolconnectionmatrix[u, v] == 1:
                    nc_graph.add_edge(self.variablelist[v],
                                      self.variablelist[u])
        edgelist_nc = nc_graph.edges()

        self.control_graph = nx.DiGraph()
        for u in range(n):
            for v in range(n):
                if controlconnectionmatrix[u, v] == 1:
                    if (self.variablelist[v], self.variablelist[u]) in \
                            edgelist_nc:
                        self.control_graph.add_edge(self.variablelist[v],
                                                    self.variablelist[u],
                                                    controlloop=0)
                    else:
                        self.control_graph.add_edge(self.variablelist[v],
                                                    self.variablelist[u],
                                                    controlloop=1)

        for node in self.control_graph.nodes():
            self.control_graph.add_node(node,
                                        nocontrolimportance=
                                        self.blendedranking_nc[node],
                                        controlimportance=
                                        self.blendedranking[node])

        plt.figure("The Controlled System")
        nx.draw_circular(self.control_graph)

    def show_all(self):
        """This method shows all figures."""
        plt.show()

    def exportogml(self):
        """This method will just export the control graphs
        to a gml file.

        """
        try:
            if self.control_graph:
                print("control_graph exists")
                nx.write_gml(self.control_graph, "control_graph.gml")
        except:
            print("control_graph does not exist")