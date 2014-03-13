# -*- coding: utf-8 -*-
"""Reports the maximum partial correlation as well as associated delay
obtained by shifting the affected variable behind the causal variable a
specified set of delays.

Also reports whether the sign changed compared to no delay.

Created on Thu Mar 13 09:44:16 2014

@author: Simon
"""

#from ranking.gaincalc import calc_partialcor_delayed
from ranking.gaincalc import create_connectionmatrix
from ranking.gaincalc import calc_max_partialcor

import json
import csv
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO)


def writecsv(filename, items, header):
    with open(filename, 'wb') as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(items)

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

# Specify desired delays in time units
delays = []
# Include first 100 2-second shifts
stageone = [val * (2.0/3600.0) for val in range(100)]
# After that, inlcude 1000 10-second shifts
stagetwo = [val * (10.0/3600.0) + stageone[-1] for val in range(1000)]
delays = list(np.hstack((stageone, stagetwo)))

for case in cases[1:2]:
    logging.info("Running case {}".format(case))
    # Get connection (adjacency) matrix
    connectionloc = os.path.join(plantdir, 'connections',
                                 caseconfig[case]['connections'])
    # Get time series data
    tags_tsdata = os.path.join(plantdir, 'data', caseconfig[case]['data'])
    # Get dataset name
    dataset = caseconfig[case]['dataset']
    # Get the variables and connection matrix
    [variables, connectionmatrix] = create_connectionmatrix(connectionloc)

    [max_val_array, max_delay_array, datastore, data_header] = \
        calc_max_partialcor(variables, connectionmatrix, tags_tsdata,
                            dataset, sampling_rate, delays)

    # Define export direcories and filenames
    datasavename = os.path.join(saveloc,
                                '{}_max_partial_data.csv'.format(case))
    value_array_savename = os.path.join(saveloc,
                                        '{}_maxcorr_array.csv'.format(case))
    delay_array_savename = os.path.join(saveloc,
                                        '{}_delay_array.csv'.format(case))
    # Write arrays to file
    np.savetxt(value_array_savename, max_val_array, delimiter=',')
    np.savetxt(delay_array_savename, max_delay_array, delimiter=',')

    # Write datastore to file
    writecsv(datasavename, datastore, data_header)
