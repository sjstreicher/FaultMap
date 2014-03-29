# -*- coding: utf-8 -*-
"""Reports the maximum partial correlation as well as associated delay
obtained by shifting the affected variable behind the causal variable a
specified set of delays.

Also reports whether the sign changed compared to no delay.

Created on Thu Mar 13 09:44:16 2014

@author: Simon
"""

from demo_setup import runsetup
from ranking.gaincalc import create_connectionmatrix
from ranking.gaincalc import estimate_delay
from ranking.formatmatrices import writecsv
# Import all test data generator functions that may be called
# via the config file
from datagen import *

from sklearn import preprocessing

import numpy as np
import os
import logging
import h5py
import jpype

logging.basicConfig(level=logging.INFO)

mode = 'test_cases'
case = 'random_2x2'

scenarios, saveloc, caseconfig, plantdir, sampling_rate, infodynamicsloc, \
    datatype = runsetup(mode, case)

# Specify delay type either as 'datapoints' or 'timevalues'
delaytype = 'datapoints'

# Specify method either as 'partial_correlation' or 'transfer_entropy'
method = 'transfer_entropy'
#method = 'partial_correlation'
logging.info("Method: " + method)

if delaytype == 'datapoints':
# Include first n sampling intervals
    delays = range(51)
elif delaytype == 'timevalues':
# Include first n 10-second shifts
    delays = [val * (10.0/3600.0) for val in range(1000)]

# Value chosen for demonstration purposes only
# Similar to subsamples in vectorselection function
size = 2000

if method == 'transfer_entropy':
    # Change location of jar to match yours:
    jarLocation = infodynamicsloc
    # Start the JVM
    # (add the "-Xmx" option with say 1024M if you get crashes
    # due to not enough memory space)
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea",
                   "-Djava.class.path=" + jarLocation)

for scenario in scenarios:
    logging.info("Running scenario {}".format(scenario))
    # Get connection (adjacency) matrix
    connectionloc = os.path.join(plantdir, 'connections',
                                 caseconfig[scenario]['connections'])
    # Get time series data
    if datatype == 'file':
        tags_tsdata = os.path.join(plantdir, 'data',
                                   caseconfig[scenario]['data'])
        # Get dataset name
        dataset = caseconfig[scenario]['dataset']
        inputdata = np.array(h5py.File(tags_tsdata, 'r')[dataset])
    elif datatype == 'function':
        tags_tsdata_gen = caseconfig['datagen']
        # TODO: Store function arguments in scenario config file
        samples = 3000
        delay = 5
        inputdata = eval(tags_tsdata_gen)(samples, delay)

    # Normalise (mean centre and variance scale) the input data
    inputdata_norm = preprocessing.scale(inputdata, axis=0)

    # Get the variables and connection matrix
    [variables, connectionmatrix] = create_connectionmatrix(connectionloc)

    [weight_array, delay_array, datastore, data_header] = \
        estimate_delay(variables, connectionmatrix, inputdata_norm,
                       sampling_rate, size, delays, delaytype, method)

    # Define export directories and filenames
    datasavename = \
        os.path.join(saveloc, 'estimate_delay',
                     '{}_{}_{}_estimate_delay_data.csv'.format(case, scenario, method))
    value_array_savename = \
        os.path.join(saveloc, 'estimate_delay',
                     '{}_{}_{}_maxweight_array.csv'.format(case, scenario, method))
    delay_array_savename = \
        os.path.join(saveloc, 'estimate_delay',
                     '{}_{}_{}_delay_array.csv'.format(case, scenario, method))

    # Write arrays to file
    np.savetxt(value_array_savename, weight_array, delimiter=',')
    np.savetxt(delay_array_savename, delay_array, delimiter=',')

    # Write datastore to file
    writecsv(datasavename, datastore, data_header)

if method == 'transfer_entropy':
    # Shutdown the JVM
    jpype.shutdownJVM()
