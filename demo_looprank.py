"""This script runs all other scripts required for calculating a ranking

@author: St. Elmo Wilken, Simon Streicher

"""

from demo_setup import runsetup
from ranking.gaincalc import create_connectionmatrix
from ranking.gaincalc import calc_partialcor_gainmatrix
from ranking.formatmatrices import rankforward, rankbackward
from ranking.formatmatrices import split_tsdata
from ranking.noderank import calc_simple_rank
from ranking.noderank import calc_blended_rank
from ranking.noderank import calc_transient_importancediffs
from ranking.noderank import plot_transient_importances
from ranking.noderank import create_importance_graph

import csv
import numpy as np
import networkx as nx

import os
import logging

logging.basicConfig(level=logging.INFO)

mode = 'test_cases'
case = 'series'

# Define the method for determining the gainmatrix
# TODO: Include this in the case configuration file
#method = 'transfer_entropy'
method = 'partial_correlation'

# Choose whether to add dummy variables or not
dummycreation = False

# Optional methods
# Do a single ranking on all data
singleranking = True
# Do a transient ranking on all data boxes
transientranking = False
# Save plots of transient rankings
transientplots = False

scenarios, saveloc, caseconfig, casedir, sampling_rate, \
    infodynamicsloc, datatype = runsetup(mode, case)

def writecsv(filename, items):
    with open(filename, 'wb') as f:
        csv.writer(f).writerows(items)

def gainrank(gainmatrix, variables, connectionmatrix,
             alpha=0.50, dummyweight=1.0):
    # TODO: The forward, backward and blended ranking will all be folded
    # into a single method, currently isolated for ease of access to
    # intermediate results
    forwardconnection, forwardgain, forwardvariablelist = \
        rankforward(variables, gainmatrix, connectionmatrix, dummyweight)
        
    backwardconnection, backwardgain, backwardvariablelist = \
        rankbackward(variables, gainmatrix, connectionmatrix, dummyweight)
        
    forwardrankingdict, forwardrankinglist = \
        calc_simple_rank(forwardgain, forwardvariablelist)
    
    backwardrankingdict, backwardrankinglist = \
        calc_simple_rank(backwardgain, backwardvariablelist)
        
    blendedrankingdict, blendedrankinglist = \
        calc_blended_rank(forwardrankingdict, backwardrankingdict,
                          variables, alpha)
    
    rankingdicts = [blendedrankingdict, forwardrankingdict, backwardrankingdict]    
    rankinglists = [blendedrankinglist, forwardrankinglist, backwardrankinglist] 
    
    return rankingdicts, rankinglists


def calc_gainmatrix(connectionmatrix, tags_tsdata, dataset,
                   method='partial_correlation'):
    """Calculates the gainmatrix from tags time series data.
    
    Can make use of either the partial correlation or transfer entropy method
    for determining weights.
    
    """
    
    if method == 'partial_correlation':
        # Get the partialcorr gainmatrix
        _, gainmatrix = \
            calc_partialcor_gainmatrix(connectionmatrix, tags_tsdata, dataset)
            
    elif method == 'transfer_entropy':
        # TODO: Implement transfer entropy weight calculation
        gainmatrix = None
    
    savename = os.path.join(saveloc, "gainmatrix.csv")
    np.savetxt(savename, gainmatrix, delimiter=',')
     
    return gainmatrix
    
    
def looprank_single(scenario, variables, connectionmatrix, gainmatrix):
    """Ranks the nodes in a network given a gainmatrix, connectionmatrix
    and a list of variables.
    
    """
    
    # TODO: Refactor some more main loop functions here
    
    rankingdicts, rankinglists = gainrank(gainmatrix, variables,
                                          connectionmatrix)                          
    
    directions = ['blended', 'forward', 'backward']                                    
    for direction, rankinglist, rankingdict in zip(directions, rankinglists, rankingdicts):
        # Save the ranking list to file        
        savename = os.path.join(saveloc, scenario + '_{}_importances.csv'.format(direction))
        writecsv(savename, rankinglist)
        # Save the graphs to file
        closedgraph, _ = \
                create_importance_graph(variables, connectionmatrix,
                                        connectionmatrix, gainmatrix,
                                        rankingdict)
        closedgraph_filename = os.path.join(saveloc,
                                            "{}_{}_closedgraph.gml".format(scenario, direction))

        nx.readwrite.write_gml(closedgraph, closedgraph_filename)
   
    logging.info("Done with single ranking")
    
    return None


def looprank_transient(scenario, samplerate, boxsize, boxnum, variables,
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
                         "{}_gainmatrix_{:03d}.csv".format(scenario, index))
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
    
    # Plotting functions
    diffplot, absplot = plot_transient_importances(variables,
                                                   transientdict,
                                                   basevaldict)
    diffplot_filename = os.path.join(saveloc,
                                     "{}_diffplot.pdf".format(scenario))
    absplot_filename = os.path.join(saveloc,
                                    "{}_absplot.pdf".format(scenario))
    diffplot.savefig(diffplot_filename)
    absplot.savefig(absplot_filename)

    logging.info("Done with transient rankings")

    return None

for scenario in scenarios:
    logging.info("Running scenario {}".format(scenario))
    if datatype == 'file':
        # Get connection (adjacency) matrix
        connectionloc = os.path.join(casedir, 'connections',
                                     caseconfig[scenario]['connections'])
        # Get the variables and closedloop connection matrix
        [variables, connectionmatrix] = create_connectionmatrix(connectionloc)
        # Get the openloop connectionmatrix
        openconnectionloc = os.path.join(casedir, 'connections',
                                 caseconfig['open_connections'])
        [_, openconnectionmatrix] = create_connectionmatrix(openconnectionloc)

        # Get time series data
        tags_tsdata = os.path.join(casedir, 'data', caseconfig[scenario]['data'])
        # Get dataset name
        dataset = caseconfig[scenario]['dataset']
        
        # Calculate the gainmatrix
        gainmatrix = calc_gainmatrix(connectionmatrix, tags_tsdata, dataset)
        
        boxnum = caseconfig[scenario]['boxnum']
        boxsize = caseconfig[scenario]['boxsize']
        
    elif datatype == 'function':
        # Get variables, connection matrix and gainmatrix
        # For the test cases the openloop connection matrix is kept identical
        # to the closedloop connection matrix
        network_gen = caseconfig[scenario]['networkgen']
        connectionmatrix, gainmatrix, variables, _ = eval(network_gen)

    logging.info("Number of tags: {}".format(len(variables)))

    if singleranking:    
        looprank_single(scenario, variables,
                        connectionmatrix, gainmatrix)
    
    if transientranking:
        [transientdict, basevaldict] = \
            looprank_transient(scenario, sampling_rate, boxsize, boxnum, variables,
                               connectionmatrix, tags_tsdata, dataset)