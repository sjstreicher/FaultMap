"""This script runs all other scripts required for calculating a ranking

@author: St. Elmo Wilken, Simon Streicher

"""

#from ranking.controlranking import LoopRanking
#from ranking.formatmatrices import FormatMatrix
from ranking.localgaincalc import create_connectionmatrix
from ranking.localgaincalc import calc_partialcor_gainmatrix
from ranking.formatmatrices import removedummyvars
from ranking.formatmatrices import addforwardscale, addbackwardscale
from ranking.controlranking import LoopRanking
import json
import csv
import numpy as np

filesloc = json.load(open('config.json'))

connection_loc = filesloc['closedloop_connections_mod']
tags_tsdata = filesloc['data_dist11_mod']

dummy_var_no = 0

# Get the variables and connectionmatrix and save to file
[variables, connectionmatrix] = \
    create_connectionmatrix(connection_loc)

# Get the correlation and partial correlation matrices
[correlationmatrix, partialcorrelationmatrix] = \
    calc_partialcor_gainmatrix(connectionmatrix, tags_tsdata)

# Remove dummy variables
# Where did these dummy variables come from?
[nodummyvariablelist, nodummygain, nodummyconnection, nodummy_nodes] = \
    removedummyvars(partialcorrelationmatrix, connectionmatrix, variables,
                    dummy_var_no)

scaledforwardconnection, scaledforwardgain, scaledforwardvariablelist = \
    addforwardscale(nodummyvariablelist, nodummygain, nodummyconnection,
                    nodummy_nodes)

scaledbackwardconnection, scaledbackwardgain, scaledbackwardvariablelist = \
    addbackwardscale(nodummyvariablelist, nodummygain, nodummyconnection,
                     nodummy_nodes)


def write_to_files(variables, connectionmatrix, correlationmatrix,
                   partialcorrelationmatrix,
                   nodummyvariablelist, nodummygain, nodummyconnection,
                   scaledforwardconnection, scaledforwardgain,
                   scaledforwardvariablelist,
                   scaledbackwardconnection, scaledbackwardgain,
                   scaledbackwardvariablelist):
    """Writes the list of variables, connectionmatrix, correlationmatrix
    as well as partial correlation matrix to file.

    """
    with open('vars.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(variables)
    np.savetxt("connectmatrix.csv", connectionmatrix, delimiter=',')
    np.savetxt("correlmat.csv", correlationmatrix, delimiter=',')
    np.savetxt("partialcorrelmat.csv", partialcorrelationmatrix, delimiter=',')

    return None

#write_to_files(variables, connectionmatrix, correlationmatrix,
#               partialcorrelationmatrix,
#               nodummyvariablelist, nodummygain, nodummyconnection,
#               scaledforwardconnection, scaledforwardgain,
#               scaledforwardvariablelist,
#               scaledbackwardconnection, scaledbackwardgain,
#               scaledbackwardvariablelist)


controlmatrix = LoopRanking(scaledforwardgain,
                            scaledforwardvariablelist,
                            scaledforwardconnection,
                            scaledbackwardgain,
                            scaledbackwardvariablelist,
                            scaledbackwardconnection,
                            nodummyvariablelist)

#controlmatrix.display_control_importances([], datamatrix.nodummyconnection)
#controlmatrix.show_all()
#controlmatrix.exportogml()