"""This script runs all other scripts required for calculating a ranking

@author: St. Elmo Wilken, Simon Streicher

"""

#from ranking.controlranking import LoopRanking
#from ranking.formatmatrices import FormatMatrix
from ranking.localgaincalc import create_connectionmatrix
from ranking.localgaincalc import calc_partialcor_gainmatrix
#from ranking.formatmatrices import removedummyvars
from ranking.formatmatrices import addforwardscale, addbackwardscale
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
# Tags time series data
tags_tsdata = filesloc['data_dist11_mod']
#Location to store all exported files
saveloc = filesloc['savelocation']

dummy_var_no = 0

# Get the variables and clsoedloop connectionmatrix
[variables, connectionmatrix] = create_connectionmatrix(connection_loc)

# Get the openloop connectionmatrix
_, openconnectionmatrix = create_connectionmatrix(openconnection_loc)

# Get the correlation and partial correlation matrices
[correlationmatrix, partialcorrelationmatrix] = \
    calc_partialcor_gainmatrix(connectionmatrix, tags_tsdata)

# Remove dummy variables - possibly deprecated function
# Where did these dummy variables come from?
#[nodummyvariablelist, nodummygain, nodummyconnection, nodummy_nodes] = \
#    removedummyvars(partialcorrelationmatrix, connectionmatrix, variables,
#                    dummy_var_no)variables, partialcorrelationmatrix, connectionmatrix,
#                    len(variables)

#scaledforwardconnection, scaledforwardgain, scaledforwardvariablelist = \
#    addforwardscale(nodummyvariablelist, nodummygain, nodummyconnection,
#                    nodummy_nodes)

scaledforwardconnection, scaledforwardgain, scaledforwardvariablelist = \
    addforwardscale(variables, partialcorrelationmatrix, connectionmatrix)

scaledbackwardconnection, scaledbackwardgain, scaledbackwardvariablelist = \
    addbackwardscale(variables, partialcorrelationmatrix, connectionmatrix)


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


# TODO: Rethink the purpose of normalising scaledforwardgain
# Results were significantly better without it
# Does this indicate something about the dummy variables
# corrupting the rankings?
forwardrank = calculate_rank(normalise_matrix(scaledforwardgain),
                             scaledforwardvariablelist)
backwardrank = calculate_rank(normalise_matrix(scaledbackwardgain),
                              scaledbackwardvariablelist)
blendedranking, slist = create_blended_ranking(forwardrank, backwardrank,
                                               variables, alpha=0.35)

closedgraph = create_importance_graph(variables, connectionmatrix,
                                      openconnectionmatrix,
                                      blendedranking)

writer = csv.writer(open(saveloc + 'importances.csv', 'wb'))
for x in slist:
    writer.writerow(x)
    print(x)
print("Done with controlled importances")

nx.write_gml(closedgraph, saveloc + "closedgraph.gml")

#plt.figure("The Controlled System")
#nx.draw_circular(importancegraph)
#plt.show()

#controlmatrix = LoopRanking(scaledforwardgain,
#                            scaledforwardvariablelist,
#                            scaledforwardconnection,
#                            scaledbackwardgain,
#                            scaledbackwardvariablelist,
#                            scaledbackwardconnection,
#                            nodummyvariablelist)

#controlmatrix.display_control_importances([], datamatrix.nodummyconnection)
#controlmatrix.show_all()
#controlmatrix.exportogml()