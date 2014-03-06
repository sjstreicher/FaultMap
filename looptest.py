"""This script runs all other scripts required for calculating a ranking

@author: St. Elmo Wilken, Simon Streicher

"""

#from ranking.controlranking import LoopRanking
#from ranking.formatmatrices import FormatMatrix
from ranking.localgaincalc import create_connectionmatrix
from ranking.localgaincalc import calc_partialcor_gainmatrix
#from ranking.formatmatrices import removedummyvars
from ranking.formatmatrices import rankforward, rankbackward
from ranking.formatmatrices import normalise_matrix
from ranking.gainrank import calculate_rank
from ranking.gainrank import create_blended_ranking
from ranking.gainrank import create_importance_graph
import json
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

filesloc = json.load(open('config.json'))
# Closedloop connection (adjacency) matrix
connection_loc = filesloc['closedloop_connections_mod']
# Openloop connection (adjacency) matrix
openconnection_loc = filesloc['openloop_connections_mod']
closedconnection_loc = filesloc['closedloop_connections_mod']
# Tags time series data
tags_tsdata = filesloc['data_dist11_mod']
#Location to store all exported files
saveloc = filesloc['savelocation']

# TODO: Include sign of action in adjacency matrix
# (determined by process knowledge).
# Time delays may influence signs in correlations(??)

# Get the variables and clsoedloop connectionmatrix
[variables, connectionmatrix] = create_connectionmatrix(connection_loc)

# Get the openloop connectionmatrix
_, openconnectionmatrix = create_connectionmatrix(openconnection_loc)
#np.savetxt(saveloc + "openconnectmatrix.csv", openconnectionmatrix,
#           delimiter=',')

_, closedconnectionmatrix = create_connectionmatrix(closedconnection_loc)

# Get the correlation and partial correlation matrices
_, gainmatrix = \
    calc_partialcor_gainmatrix(connectionmatrix, tags_tsdata)
np.savetxt(saveloc + "gainmatrix.csv", gainmatrix,
           delimiter=',')

forwardconnection, forwardgain, forwardvariablelist = \
    rankforward(variables, gainmatrix, connectionmatrix, 0.01)
backwardconnection, backwardgain, backwardvariablelist = \
    rankbackward(variables, gainmatrix, connectionmatrix, 0.01)

forwardrank = calculate_rank(forwardgain, forwardvariablelist)
backwardrank = calculate_rank(backwardgain, backwardvariablelist)
blendedranking, slist = create_blended_ranking(forwardrank, backwardrank,
                                               variables, alpha=0.35)

closedgraph, opengraph = create_importance_graph(variables,
                                                 closedconnectionmatrix.T,
                                                 openconnectionmatrix.T,
                                                 gainmatrix,
                                                 blendedranking)
print "Number of tags: ", len(variables)

with open(saveloc + 'importances.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for x in slist:
        writer.writerow(x)
        print(x)
print("Done with controlled importances")


nx.write_gml(closedgraph, saveloc + "closedgraph.gml")
nx.write_gml(opengraph, saveloc + "opengraph.gml")

#plt.figure("Closedloop System")
#nx.draw(closedgraph)
#plt.show()
#
#plt.figure("Openloop System")


#def write_to_files(variables, connectionmatrix, correlationmatrix,
#                   partialcorrelationmatrix,
#                   scaledforwardconnection, scaledforwardgain,
#                   scaledforwardvariablelist,
#                   scaledbackwardconnection, scaledbackwardgain,
#                   scaledbackwardvariablelist):
#    """Writes the list of variables, connectionmatrix, correlationmatrix
#    as well as partial correlation matrix to file.
#
#    """
#    with open('vars.csv', 'wb') as csvfile:
#        writer = csv.writer(csvfile, delimiter=',')
#        writer.writerow(variables)
#    np.savetxt("connectmatrix.csv", connectionmatrix, delimiter=',')
#    np.savetxt("correlmat.csv", correlationmatrix, delimiter=',')
#    np.savetxt("partialcorrelmat.csv", partialcorrelationmatrix, delimiter=',')
#    np.savetxt("forwardconnection.csv", scaledforwardconnection, delimiter=',')
#
#write_to_files(variables, connectionmatrix, correlationmatrix,
#               partialcorrelationmatrix,
#               scaledforwardconnection, scaledforwardgain,
#               scaledforwardvariablelist,
#               scaledbackwardconnection, scaledbackwardgain,
#               scaledbackwardvariablelist)