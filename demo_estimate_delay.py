# -*- coding: utf-8 -*-
"""Reports the maximum partial correlation as well as associated delay
obtained by shifting the affected variable behind the causal variable a
specified set of delays.

Also reports whether the sign changed compared to no delay.

Created on Thu Mar 13 09:44:16 2014

@author: Simon
"""

from sklearn import preprocessing

from demo_setup import runsetup
from ranking.gaincalc import create_connectionmatrix
from ranking.gaincalc import estimate_delay

# Import all test data generator functions that may be called
# via the config file
from datagen import *

import csv
import numpy as np
import os
import logging
import h5py
import jpype
import json


def writecsv(filename, items, header):
    with open(filename, 'wb') as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(items)

logging.basicConfig(level=logging.INFO)


def weightcalc(mode, case, writeoutput=False):

    saveloc, casedir, infodynamicsloc = runsetup(mode, case)

    # Load case config file
    caseconfig = json.load(open(os.path.join(casedir, case + '.json')))

    # Get scenarios
    scenarios = caseconfig['scenarios']
    # Get sampling rate
    sampling_rate = caseconfig['sampling_rate']
    # Get data type
    datatype = caseconfig['datatype']
    # Get delay type
    delaytype = caseconfig['delaytype']
    # Get methods
    methods = caseconfig['methods']
    # Get size of sample vectors for tests
    # Must be smaller than number of samples generated
    testsize = caseconfig['testsize']
    # Get number of delays to test
    test_delays = caseconfig['test_delays']

    if delaytype == 'datapoints':
    # Include first n sampling intervals
        delays = range(test_delays + 1)
    elif delaytype == 'timevalues':
    # Include first n 10-second shifts
        delays = [val * (10.0/3600.0) for val in range(test_delays + 1)]

    # Start JVM if required
    if 'transfer_entropy' in methods:
        # Change location of jar to match yours:
        jarLocation = infodynamicsloc
        # Start the JVM
        # (add the "-Xmx" option with say 1024M if you get crashes
        # due to not enough memory space)
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea",
                       "-Djava.class.path=" + jarLocation)

    for scenario in scenarios:
        logging.info("Running scenario {}".format(scenario))

        # Get time series data
        if datatype == 'file':
            tags_tsdata = os.path.join(plantdir, 'data',
                                       caseconfig[scenario]['data'])
            # Get connection (adjacency) matrix
            connectionloc = os.path.join(plantdir, 'connections',
                                         caseconfig[scenario]['connections'])
            # Get dataset name
            dataset = caseconfig[scenario]['dataset']
            # Get inputdata
            inputdata = np.array(h5py.File(tags_tsdata, 'r')[dataset])
            # Get the variables and connection matrix
            [variables, connectionmatrix] = \
                create_connectionmatrix(connectionloc)

        elif datatype == 'function':
            tags_tsdata_gen = caseconfig[scenario]['datagen']
            connectionloc = caseconfig[scenario]['connections']
            # TODO: Store function arguments in scenario config file
            samples = caseconfig['gensamples']
            delay = caseconfig['delay']
            # Get inputdata
            inputdata = eval(tags_tsdata_gen)(samples, delay)
            # Get the variables and connection matrix
            [variables, connectionmatrix] = eval(connectionloc)()

        # Normalise (mean centre and variance scale) the input data
        inputdata_norm = preprocessing.scale(inputdata, axis=0)

        for method in methods:
            logging.info("Method: " + method)

            [weight_array, delay_array, datastore, data_header] = \
                estimate_delay(variables, connectionmatrix, inputdata_norm,
                               sampling_rate, testsize, delays, delaytype,
                               method)

            if writeoutput:
                # Define export directories and filenames
                datasavename = \
                    os.path.join(saveloc, 'estimate_delay',
                                 '{}_{}_{}_estimate_delay_data.csv'
                                 .format(case, scenario, method))
                value_array_savename = \
                    os.path.join(saveloc, 'estimate_delay',
                                 '{}_{}_{}_maxweight_array.csv'
                                 .format(case, scenario, method))
                delay_array_savename = \
                    os.path.join(saveloc, 'estimate_delay',
                                 '{}_{}_{}_delay_array.csv'
                                 .format(case, scenario, method))

                # Write arrays to file
                np.savetxt(value_array_savename, weight_array, delimiter=',')
                np.savetxt(delay_array_savename, delay_array, delimiter=',')

                # Write datastore to file
                writecsv(datasavename, datastore, data_header)

    # Stop JVM if required
    if 'transfer_entropy' in methods:
        # Shutdown the JVM
        jpype.shutdownJVM()

if __name__ == '__main__':
    mode = 'test_cases'
    case = 'weightcalc_tests'
    weightcalc(mode, case, False)
