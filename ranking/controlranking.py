"""This method is imported by looptest

@author: St. Elmo Wilken, Simon Streicher

"""

#from visualise import visualiseOpenLoopSystem
from gainrank import GainRanking
from numpy import array, asarray
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
from formatmatrices import normalise_matrix
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
        """This constructor creates graphs with associated node importances
        based on:
        1) Local gain information
        2) Partial correlation data

        """

        self.forwardgain = GainRanking(normalise_matrix(fgainmatrix),
                                       fvariablenames)
        self.backwardgain = GainRanking(normalise_matrix(bgainmatrix),
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


    def display_control_importances(self, nocontrolconnectionmatrix,
                                    controlconnectionmatrix):
        """Generates a graph containing the
        connectivity and importance of the system being displayed.
        Edge Attribute: color for control connection
        Node Attribute: node importance

        """
        nc_graph = nx.DiGraph()

        for u, v in nocontrolconnectionmatrix.nonzero().T:
            nc_graph.add_edge(self.variablelist[v], self.variablelist[u])
        edgelist_nc = nc_graph.edges()

        self.control_graph = nx.DiGraph()
        for u, v in controlconnectionmatrix.nonzero().T:
            newedge = (self.variablelist[v], self.variablelist[u])
            self.control_graph.add_edge(*newedge, controlloop=int(newedge in edgelist_nc))

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
