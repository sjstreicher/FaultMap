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

import json
import csv
import numpy as np

import os
import logging
logging.basicConfig(level=logging.INFO)

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

def writecsv(filename, items):
    with open(filename, 'wb') as f:
        csv.writer(f).writerows(items)


def gainrank(gainmatrix):
    # TODO: The forward, backward and blended ranking will all be folded
    # into a single method, currently isolated for ease of access to
    # intermediate results
    forwardconnection, forwardgain, forwardvariablelist = rankforward(variables, gainmatrix, connectionmatrix, 0.01)
    backwardconnection, backwardgain, backwardvariablelist = rankbackward(variables, gainmatrix, connectionmatrix, 0.01)
    forwardrank = calculate_rank(forwardgain, forwardvariablelist)
    backwardrank = calculate_rank(backwardgain, backwardvariablelist)
    blendedranking, slist = create_blended_ranking(forwardrank, backwardrank, 
        variables, alpha=0.35)
    return blendedranking, slist


def looprank_single(case):
    # Get the correlation and partial correlation matrices
    _, gainmatrix = \
        calc_partialcor_gainmatrix(connectionmatrix, tags_tsdata, dataset)
    savename = os.path.join(saveloc, "gainmatrix.csv")
    np.savetxt(savename, gainmatrix, delimiter=',')
    
    _, slist = gainrank(gainmatrix)
    
    savename = os.path.join(saveloc, case + '_importances.csv')
    writecsv(savename, slist)
    
    logging.info("Done with single ranking")


def looprank_transient(case, samplerate, boxsize, boxnum):
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
        gain_filename = os.path.join(saveloc, "{}_gainmatrix_{:03d}.csv".format(case, index))
        np.savetxt(gain_filename, gainmatrix, delimiter=',')

        blendedranking, slist = gainrank(gainmatrix)
        rankinglists.append(slist)
        savename = os.path.join(saveloc, 'importances_{:03d}.csv'.format(index))
        writecsv(savename, slist)

        rankingdicts.append(blendedranking)
    
    logging.info("Done with transient rankings")
    
    transientdict, basevaldict = \
        calc_transient_importancediffs(rankingdicts, variables)

for case in cases:
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
    
    looprank_single(case)
    looprank_transient(case, sampling_rate, boxsize, boxnum)
    
    
