"""This script runs all other scripts required for calculating a ranking

@author: St. Elmo Wilken, Simon Streicher

"""

from ranking.gaincalc import create_connectionmatrix
from ranking.gaincalc import calc_partialcor_gainmatrix
from ranking.formatmatrices import rankforward, rankbackward
from ranking.formatmatrices import split_tsdata
from ranking.noderank import calculate_rank
from ranking.noderank import create_blended_ranking
from ranking.noderank import calc_transient_importancediffs
from ranking.noderank import plot_transient_importances
from ranking.noderank import create_importance_graph

import json
import csv
import numpy as np
import networkx as nx

import os
import logging

logging.basicConfig(level=logging.INFO)

# Optional methods
# Save plots of transient rankings
transientplots = True
importancegraph = True
# Load directories config file
dirs = json.load(open('config.json'))
# Get data and preferred export directories from directories config file
dataloc = os.path.expanduser(dirs['dataloc'])
saveloc = os.path.expanduser(dirs['saveloc'])
# Define plant and case names to run
plant = 'tennessee_eastman'
# Define plant data directory
plantdir = os.path.join(dataloc, 'plants', plant)
cases = ['dist11_closedloop', 'dist11_closedloop_pressup', 'dist11_full',
         'presstep_closedloop', 'presstep_full']
# Load plant config file
caseconfig = json.load(open(os.path.join(plantdir, plant + '.json')))
# Get sampling rate
sampling_rate = caseconfig['sampling_rate']
openconnectionloc = os.path.join(plantdir, 'connections',
                                 caseconfig['open_connections'])
[_, openconnectionmatrix] = create_connectionmatrix(openconnectionloc)


def writecsv(filename, items):
    with open(filename, 'wb') as f:
        csv.writer(f).writerows(items)


def gainrank(gainmatrix, variables, connectionmatrix):
    # TODO: The forward, backward and blended ranking will all be folded
    # into a single method, currently isolated for ease of access to
    # intermediate results
    forwardconnection, forwardgain, forwardvariablelist = \
        rankforward(variables, gainmatrix, connectionmatrix, 0.01)
    backwardconnection, backwardgain, backwardvariablelist = \
        rankbackward(variables, gainmatrix, connectionmatrix, 0.01)
    forwardrank = calculate_rank(forwardgain, forwardvariablelist)
    backwardrank = calculate_rank(backwardgain, backwardvariablelist)
    rankingdict, slist = create_blended_ranking(forwardrank, backwardrank,
                                                variables, alpha=0.35)
    return rankingdict, slist


def looprank_single(case, variables, connectionmatrix, tags_tsdata, dataset):
    # Get the correlation and partial correlation matrices
    _, gainmatrix = \
        calc_partialcor_gainmatrix(connectionmatrix, tags_tsdata, dataset)
    savename = os.path.join(saveloc, "gainmatrix.csv")
    np.savetxt(savename, gainmatrix, delimiter=',')

    rankingdict, rankinglist = gainrank(gainmatrix, variables,
                                        connectionmatrix)

    savename = os.path.join(saveloc, case + '_importances.csv')
    writecsv(savename, rankinglist)
    logging.info("Done with single ranking")

    return gainmatrix, rankingdict


def looprank_transient(case, samplerate, boxsize, boxnum, variables,
                       connectionmatrix, tags_tsdata, dataset):
    # Split the tags_tsdata into sets (boxes) useful for calculating
    # transient correlations
    boxes = split_tsdata(tags_tsdata, dataset, samplerate,
                         boxsize, boxnum)
    # Calculate gain matrix for each box
    gainmatrices = [calc_partialcor_gainmatrix(connectionmatrix, box,
                                               dataset)[1]
                    for box in boxes]
    rankinglists = []
    rankingdicts = []
    for index, gainmatrix in enumerate(gainmatrices):
        # Store the gainmatrix
        gain_filename = \
            os.path.join(saveloc,
                         "{}_gainmatrix_{:03d}.csv".format(case, index))
        np.savetxt(gain_filename, gainmatrix, delimiter=',')

        blendedranking, slist = gainrank(gainmatrix, variables,
                                         connectionmatrix)
        rankinglists.append(slist)

        savename = os.path.join(saveloc,
                                'importances_{:03d}.csv'.format(index))
        writecsv(savename, slist)

        rankingdicts.append(blendedranking)

    transientdict, basevaldict = \
        calc_transient_importancediffs(rankingdicts, variables)

    logging.info("Done with transient rankings")

    return transientdict, basevaldict

# Only do a single case in the demo for coverage analysis purposes
for case in [cases[1]]:
    # Get connection (adjacency) matrix
    logging.info("Running case {}".format(case))
    connectionloc = os.path.join(plantdir, 'connections',
                                 caseconfig[case]['connections'])

    # Get time series data
    tags_tsdata = os.path.join(plantdir, 'data', caseconfig[case]['data'])
    # Get dataset name
    dataset = caseconfig[case]['dataset']
    # Get the variables and connection matrix
    [variables, connectionmatrix] = create_connectionmatrix(connectionloc)

    logging.info("Number of tags: {}".format(len(variables)))
    boxnum = caseconfig[case]['boxnum']
    boxsize = caseconfig[case]['boxsize']

    gainmatrix, rankingdict = looprank_single(case, variables,
                                              connectionmatrix,
                                              tags_tsdata, dataset)

    [transientdict, basevaldict] = \
        looprank_transient(case, sampling_rate, boxsize, boxnum, variables,
                           connectionmatrix, tags_tsdata, dataset)

    if transientplots:
        diffplot, absplot = plot_transient_importances(variables,
                                                       transientdict,
                                                       basevaldict)
        diffplot_filename = os.path.join(saveloc,
                                         "{}_diffplot.pdf".format(case))
        absplot_filename = os.path.join(saveloc,
                                        "{}_absplot.pdf".format(case))
        diffplot.savefig(diffplot_filename)
        absplot.savefig(absplot_filename)

    if importancegraph:
        closedgraph, opengraph = \
            create_importance_graph(variables, connectionmatrix,
                                    openconnectionmatrix, gainmatrix,
                                    rankingdict)
        closedgraph_filename = os.path.join(saveloc,
                                            "{}_closedgraph.gml".format(case))
        opengraph_filename = os.path.join(saveloc,
                                          "{}_opengraph.gml".format(case))

        nx.readwrite.write_gml(closedgraph, closedgraph_filename)
        nx.readwrite.write_gml(opengraph, opengraph_filename)
