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
from ranking.gaincalc import calc_max_partialcorr_delay

from sklearn import preprocessing

import json
import csv
import numpy as np
import os
import logging
import h5py
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

# Specify delay type either as 'datapoints' or 'timevalues'
delaytype = 'datapoints'
# Specify desired delays in time units
delays = []

if delaytype == 'datapoints':
    # Include first n sampling intervals
    delays = range(3000)
elif delaytype == 'timevalues':
# Include first n 10-second shifts
    delays = [val * (10.0/3600.0) for val in range(1000)]

size = 50000

# Only do a single case in the demo for coverage analysis purposes
for case in [cases[1]]:
    logging.info("Running case {}".format(case))
    # Get connection (adjacency) matrix
    connectionloc = os.path.join(plantdir, 'connections',
                                 caseconfig[case]['connections'])
    # Get time series data
    tags_tsdata = os.path.join(plantdir, 'data', caseconfig[case]['data'])
    # Get dataset name
    dataset = caseconfig[case]['dataset']
    inputdata = np.array(h5py.File(tags_tsdata, 'r')[dataset])
    # Normalise (mean centre and variance scale) the input data
    inputdata_norm = preprocessing.scale(inputdata)

    # Get the variables and connection matrix
    [variables, connectionmatrix] = create_connectionmatrix(connectionloc)

    [corr_array, delay_array, datastore, data_header] = \
        calc_max_partialcorr_delay(variables, connectionmatrix, inputdata_norm,
                                   dataset, sampling_rate, delays,
                                   size, delaytype)

    # Define export direcories and filenames
    datasavename = os.path.join(saveloc, '/max_partialcorr/',
                                '{}_max_partial_data.csv'.format(case))
    value_array_savename = os.path.join(saveloc, '/max_partialcorr/',
                                        '{}_maxcorr_array.csv'.format(case))
    delay_array_savename = os.path.join(saveloc, '/max_partialcorr/'
                                        '{}_delay_array.csv'.format(case))
    # Write arrays to file
    np.savetxt(value_array_savename, corr_array, delimiter=',')
    np.savetxt(delay_array_savename, delay_array, delimiter=',')

    # Write datastore to file
    writecsv(datasavename, datastore, data_header)
