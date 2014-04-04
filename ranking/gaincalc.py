"""This method is imported by formatmatrices

@author: St. Elmo Wilken, Simon Streicher

"""

import os
import csv
import numpy as np
import h5py
import logging
import jpype
import json
from config_setup import ensure_existance

from sklearn import preprocessing
from transentropy import calc_infodynamics_te as te_infodyns
from transentropy import setup_infodynamics_te as te_setup
from config_setup import runsetup

# Import all test data geneartors that may be called
from datagen import *


def create_connectionmatrix(connection_loc):
    """This method imports the connection scheme for the data.
    The format should be:
    empty space, var1, var2, etc... (first row)
    var1, value, value, value, etc... (second row)
    var2, value, value, value, etc... (third row)
    etc...

    Value is 1 if column variable points to row variable
    (causal relationship)
    Value is 0 otherwise

    This method also stores the names of all the variables in the
    connection matrix.
    It is important that the order of the variables in the
    connection matrix match those in the data matrix.

    """
    with open(connection_loc) as f:
        variables = csv.reader(f).next()[1:]
        connectionmatrix = np.genfromtxt(f, delimiter=',')[:, 1:]

    return variables, connectionmatrix


def calc_partialcor_gainmatrix(connectionmatrix, tags_tsdata, dataset):
    """Calculates the local gains in terms of the partial (Pearson's)
    correlation between the variables.

    connectionmatrix is the adjacency matrix

    tags_tsdata contains the time series data for the tags with variables
    in colums and sampling instances in rows

    """
    if isinstance(tags_tsdata, np.ndarray):
        inputdata = tags_tsdata
    else:
        inputdata = np.array(h5py.File(tags_tsdata, 'r')[dataset])
#    print "Total number of data points: ", inputdata.size
    # Calculate correlation matrix
    correlationmatrix = np.corrcoef(inputdata.T)
    # Calculate partial correlation matrix
    p_matrix = np.linalg.inv(correlationmatrix)
    d = p_matrix.diagonal()
    partialcorrelationmatrix = \
        np.where(connectionmatrix, -p_matrix/np.abs(np.sqrt(np.outer(d, d))),
                 0)

    return correlationmatrix, partialcorrelationmatrix


def partialcorr_reporting(weightlist, actual_delays, weight_array, delay_array,
                          threshcorr, threshdir,
                          affectedvarindex, causevarindex, datastore,
                          causevar, affectedvar):

    maxval = max(weightlist)
    minval = min(weightlist)
    if (maxval + minval) >= 0:
        delay_index = weightlist.index(maxval)
        maxcorr = maxval
    else:
        delay_index = weightlist.index(minval)
        maxcorr = minval
    # Bauer2008 eq. 4
    maxcorr_abs = max(maxval, abs(minval))
    bestdelay = actual_delays[delay_index]
    directionindex = 2 * (abs(maxval + minval) /
                          (maxval + abs(minval)))
    weight_array[affectedvarindex, causevarindex] = maxcorr
    delay_array[affectedvarindex, causevarindex] = bestdelay

    if (weightlist[0] / weightlist[delay_index]) >= 0:
        signchange = False
    else:
        signchange = True

    if maxcorr_abs >= threshcorr:
        corrthreshpass = True
    else:
        corrthreshpass = False
        # TODO: Don't use the shifted correlation

    if directionindex >= threshdir:
        dirthreshpass = True
    else:
        dirthreshpass = False
        # TODO: Don't use the shifted correlation

    dataline = [causevar, affectedvar, str(weightlist[0]),
                maxcorr, str(bestdelay), str(delay_index),
                signchange, corrthreshpass, dirthreshpass, directionindex]

    datastore.append(dataline)
    logging.info("Maximum correlation value: " + str(maxval))
    logging.info("Minimum correlation value: " + str(minval))
    logging.info("The maximum correlation between " + causevar +
                 " and " + affectedvar + " is: " + str(maxcorr))
    logging.info("The corresponding delay is: " +
                 str(bestdelay))
    logging.info("The correlation with no delay is: "
                 + str(weightlist[0]))
    logging.info("Correlation threshold passed: " +
                 str(corrthreshpass))
    logging.info("Directionality value: " + str(directionindex))
    logging.info("Directionality threshold passed: " +
                 str(dirthreshpass))

    return weight_array, delay_array, datastore


def transent_reporting(weightlist, actual_delays, weight_array, delay_array,
                       threshent, affectedvarindex, causevarindex,
                       datastore, causevar, affectedvar):

    maxval = max(weightlist)
    weight_array[affectedvarindex, causevarindex] = maxval

    delay_index = weightlist.index(maxval)
    bestdelay = actual_delays[delay_index]
    delay_array[affectedvarindex, causevarindex] = bestdelay

    if maxval >= threshent:
        threshpass = True
    else:
        threshpass = False

    dataline = [causevar, affectedvar, str(weightlist[0]),
                maxval, str(bestdelay), str(delay_index),
                threshpass]
    datastore.append(dataline)

    logging.info("The maximum TE between " + causevar +
                 " and " + affectedvar + " is: " + str(maxval))
    logging.info("The corresponding delay is: " +
                 str(bestdelay))
    logging.info("The TE with no delay is: "
                 + str(weightlist[0]))
#    logging.info("TE threshold passed: " +
#                 str(threshpass))

    return weight_array, delay_array, datastore


def estimate_delay(variables, connectionmatrix, inputdata,
                   sampling_rate, size, delays, delaytype, method):
    """Determines the maximum weight between two variables by searching through
    a specified set of delays.

    method can be either 'partial_correlation' or 'transfer_entropy'

    size refers to the number of elements of two time series data vectors used
    It is kept constant so as to eliminate any effect that different
    vector length might have on partial correlation

    inputdata should be normalised (mean centered and variance scaled)

    """
    weight_array = np.empty((len(variables), len(variables)))
    delay_array = np.empty((len(variables), len(variables)))
    weight_array[:] = np.NAN
    delay_array[:] = np.NAN

    # Normalise inputdata to be safe
    inputdata = preprocessing.scale(inputdata, axis=0)

    if method == 'partial_correlation':

        threshcorr = (1.85*(size**(-0.41))) + (2.37*(size**(-0.53)))
        threshdir = 0.46*(size**(-0.16))

        logging.info("Directionality threshold: " + str(threshdir))
        logging.info("Correlation threshold: " + str(threshcorr))

        data_header = ['causevar', 'affectedvar', 'base_corr',
                       'max_corr', 'max_delay', 'max_index',
                       'signchange', 'corrthreshpass',
                       'dirrthreshpass', 'dirval']

    elif method == 'transfer_entropy':
        # TODO: Get transfer entropy threshold from Bauer2005
        threshent = 0.0

        data_header = ['causevar', 'affectedvar', 'base_ent',
                       'max_ent', 'max_delay', 'max_index', 'threshpass']

    datastore = []
    if delaytype == 'datapoints':
        actual_delays = [delay * sampling_rate for delay in delays]
        sample_delay = delays
    elif delaytype == 'timevalues':
        actual_delays = [int(round(delay/sampling_rate)) * sampling_rate
                         for delay in delays]
        sample_delay = [int(round(delay/sampling_rate))
                        for delay in delays]

    for causevarindex, causevar in enumerate(variables):
        logging.info("Analysing effect of: " + causevar)
        for affectedvarindex, affectedvar in enumerate(variables):
            if not(connectionmatrix[affectedvarindex, causevarindex] == 0):
                weightlist = []
                for delay in sample_delay:

                    if delay == 0:
                        causeoffset = None
                    else:
                        causeoffset = -delay

                    causevardata = \
                        inputdata[:, causevarindex][-(size+delay):causeoffset]
                    affectedvardata = \
                        inputdata[:, affectedvarindex][-(size):]

                    if method == 'partial_correlation':
                        corrval = \
                            np.corrcoef(causevardata.T,
                                        affectedvardata.T)[1, 0]
                        weightlist.append(corrval)

                    elif method == 'transfer_entropy':
                        # Setup Java class for infodynamics toolkit
                        teCalc = te_setup()
                        transent = \
                            te_infodyns(teCalc,
                                        affectedvardata.T, causevardata.T)
                        weightlist.append(transent)
                        # Delete teCalc class in order to allow
                        # garbage data to be removed
                        # TODO: Find a method that works
                        # This does not have any effect at all
#                        transent = None
#                        del transent
#                        teCalc = None
#                        del teCalc
#                        jpype.java.lang.System.gc()

                if method == 'partial_correlation':
                    [weight_array, delay_array, datastore] = \
                        partialcorr_reporting(weightlist, actual_delays,
                                              weight_array, delay_array,
                                              threshcorr, threshdir,
                                              affectedvarindex, causevarindex,
                                              datastore, causevar, affectedvar)
                elif method == 'transfer_entropy':
                    [weight_array, delay_array, datastore] = \
                        transent_reporting(weightlist, actual_delays,
                                           weight_array, delay_array,
                                           threshent, affectedvarindex,
                                           causevarindex, datastore, causevar,
                                           affectedvar)
            else:
                weight_array[affectedvarindex, causevarindex] = np.NAN
                delay_array[affectedvarindex, causevarindex] = np.NAN

    return weight_array, delay_array, datastore, data_header


def writecsv_weightcalc(filename, items, header):
    """CSV writer customized for use in weightcalc function."""

    with open(filename, 'wb') as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(items)


def weightcalc(mode, case, writeoutput=False):
    """Reports the maximum partial correlation as well as associated delay
    obtained by shifting the affected variable behind the causal variable a
    specified set of delays.

    Also reports whether the sign changed compared to no delay.

    """

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
        if not jpype.isJVMStarted():
            # Start the JVM
            # (add the "-Xmx" option with say 1024M if you get crashes
            # due to not enough memory space)
#            jpype.startJVM(jpype.getDefaultJVMPath(), "-ea",
#                           "-Djava.class.path=" + infodynamicsloc)
            jpype.startJVM(jpype.getDefaultJVMPath(),
                           "-XX:+HeapDumpOnOutOfMemoryError",
                           "-XX:HeapDumpPath=C:/Repos/LoopRank/dumps",
                           "-verbose:gc",
                           "-Xms4M",
                           "-Xmx32M",
                           "-ea",
                           "-Djava.class.path=" + infodynamicsloc)

    for scenario in scenarios:
        logging.info("Running scenario {}".format(scenario))

        # Get time series data
        if datatype == 'file':
            # Get time series data
            tags_tsdata = os.path.join(casedir, 'data',
                                       caseconfig[scenario]['data'])
            # Get connection (adjacency) matrix
            connectionloc = os.path.join(casedir, 'connections',
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
                weightdir = ensure_existance(os.path.join(saveloc, 'weightcalc'),
                                             make=True)
                filename_template = os.path.join(weightdir, '{}_{}_{}_{}.csv')

                def filename(name):
                    return filename_template.format(case, scenario, method, name)

                # Write arrays to file
                np.savetxt(filename('maxweight_array'), weight_array, delimiter=',')
                np.savetxt(filename('delay_array'), delay_array, delimiter=',')

                # Write datastore to file
                writecsv_weightcalc(filename('weightcalc_data'), datastore, data_header)
