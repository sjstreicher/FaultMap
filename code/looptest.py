"""This class is used to run controlranking

@author: St. Elmo Wilken, Simon Streicher

"""

from controlranking import LoopRanking
from formatmatrices import FormatMatrix
from numpy import array, transpose, arange, empty
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter

datamatrix = FormatMatrix("connectionsTEcontrol.csv", "data.csv", 0, 0,
                          partialcorrelation=True)
controlmatrix = LoopRanking(datamatrix.scaledforwardgain,
                            datamatrix.scaledforwardvariablelist,
                            datamatrix.scaledforwardconnection,
                            datamatrix.scaledbackwardgain,
                            datamatrix.scaledbackwardvariablelist,
                            datamatrix.scaledbackwardconnection,
                            datamatrix.nodummyvariablelist, [])

#controlmatrix.display_control_importances([], datamatrix.nodummyconnection)
#controlmatrix.show_all()
#controlmatrix.exportogml()