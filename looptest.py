"""This script runs all other scripts required for calculating a ranking

@author: St. Elmo Wilken, Simon Streicher

"""


from ranking.localgaincalc import create_connectionmatrix
from ranking.localgaincalc import calc_partialcor_gainmatrix
from ranking.formatmatrices import rankforward, rankbackward
from ranking.gainrank import calculate_rank
from ranking.gainrank import create_blended_ranking
from ranking.gainrank import create_importance_graph
import json
import csv
import numpy as np

filesloc = json.load(open('config.json'))
# Closedloop connection (adjacency) matrix
connection_loc = filesloc['closedloop_connections_mod_pressup']
# Openloop connection (adjacency) matrix
openconnection_loc = filesloc['openloop_connections_mod']
closedconnection_loc = filesloc['closedloop_connections_mod']
# Tags time series data
datasetname = 'data_dist11_mod'
tags_tsdata = filesloc[datasetname]
#Location to store all exported files
saveloc = filesloc['savelocation']

# TODO: Include sign of action in adjacency matrix
# (determined by process knowledge).
# Time delays may influence signs in correlations.
# Look at sign-retaining time delay estimation done by Labuschagne2008.

# Get the variables and clsoedloop connectionmatrix
[variables, connectionmatrix] = create_connectionmatrix(connection_loc)

# Get the openloop connectionmatrix
_, openconnectionmatrix = create_connectionmatrix(openconnection_loc)

_, closedconnectionmatrix = create_connectionmatrix(closedconnection_loc)


# Get the correlation and partial correlation matrices
_, gainmatrix = \
    calc_partialcor_gainmatrix(connectionmatrix, tags_tsdata, datasetname)
np.savetxt(saveloc + "gainmatrix.csv", gainmatrix,
           delimiter=',')

forwardconnection, forwardgain, forwardvariablelist = \
    rankforward(variables, gainmatrix, connectionmatrix, 0.01)
backwardconnection, backwardgain, backwardvariablelist = \
    rankbackward(variables, gainmatrix, connectionmatrix, 0.01)

forwardrank = calculate_rank(forwardgain, forwardvariablelist)
backwardrank = calculate_rank(backwardgain, backwardvariablelist)
# TODO: Why is there no difference?
# Does it have to do with symmetry of partial correlation matrix?
blendedranking, slist = create_blended_ranking(forwardrank, backwardrank,
                                               variables, alpha=0.35)

allgraph, _ = create_importance_graph(variables,
                                             connectionmatrix.T,
                                             connectionmatrix.T,
                                             gainmatrix,
                                             blendedranking)
print "Number of tags: ", len(variables)

with open(saveloc + 'importances.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for x in slist:
        writer.writerow(x)
        print(x)
print("Done with controlled importances")