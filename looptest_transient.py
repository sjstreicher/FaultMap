"""This script runs all other scripts required for calculating a ranking

@author: St. Elmo Wilken, Simon Streicher

"""

#from ranking.controlranking import LoopRanking
#from ranking.formatmatrices import FormatMatrix
from ranking.localgaincalc import create_connectionmatrix
from ranking.localgaincalc import calc_partialcor_gainmatrix
#from ranking.formatmatrices import removedummyvars
from ranking.formatmatrices import split_tsdata
from ranking.formatmatrices import rankforward, rankbackward
from ranking.gainrank import calculate_rank
from ranking.gainrank import create_blended_ranking
from ranking.gainrank import calc_transient_importancediffs
from ranking.gainrank import create_importance_graph
import json
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

# Define sample rate in terms of base time unit
samplerate = 5e-4

# TODO: Include sign of action in adjacency matrix
# (determined by process knowledge).
# Time delays may influence signs in correlations(??)

# Get the variables and closedloop connectionmatrix
[variables, connectionmatrix] = create_connectionmatrix(connection_loc)
# Get the openloop connectionmatrix
_, openconnectionmatrix = create_connectionmatrix(openconnection_loc)
_, closedconnectionmatrix = create_connectionmatrix(closedconnection_loc)

# Split the tags_tsdata into sets useful for calculating
# transient correlations.
boxes = split_tsdata(tags_tsdata, datasetname, samplerate, 2, 10)

# Calculate gain matrix for each box
gainmatrices = [calc_partialcor_gainmatrix(connectionmatrix, box,
                                           datasetname)[1]
                for box in boxes]

rankinglists = []
rankingdicts = []
for index, gainmatrix in enumerate(gainmatrices):
    # Store the gainmatrix
    np.savetxt(saveloc + "gainmatrix_" + str(index) + ".csv", gainmatrix,
               delimiter=',')
    # Get everything needed to calculate slist

    forwardconnection, forwardgain, forwardvariablelist = \
        rankforward(variables, gainmatrix, connectionmatrix, 0.01)
    backwardconnection, backwardgain, backwardvariablelist = \
        rankbackward(variables, gainmatrix, connectionmatrix, 0.01)

    forwardrank = calculate_rank(forwardgain, forwardvariablelist)
    backwardrank = calculate_rank(backwardgain, backwardvariablelist)
    # Why is there no difference?
    blendedranking, slist = create_blended_ranking(forwardrank, backwardrank,
                                                   variables, alpha=0.35)
    rankinglists.append(slist)
    with open(saveloc + 'importances_' + str(index) + '.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for x in slist:
            writer.writerow(x)
#            print(x)
    rankingdicts.append(blendedranking)

print("Done with controlled importances")

transientdict, basevaldict = \
    calc_transient_importancediffs(rankingdicts, variables)