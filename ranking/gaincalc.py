"""This method is imported by formatmatrices

@author: St. Elmo Wilken, Simon Streicher

"""

import csv
import numpy as np
import h5py
import logging

from sklearn import preprocessing
from transentropy import calc_infodynamics_te as te_infodyns
from transentropy import setup_infodynamics_te as te_setup


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
    # +1 needed due to the way delays defined in atx function - revise
    bestdelay = actual_delays[delay_index+1]
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
        # Don't use the shifted correlation

    if directionindex >= threshdir:
        dirthreshpass = True
    else:
        dirthreshpass = False
        # Don't use the shifted correlatio

    dataline = [causevar, affectedvar, str(weightlist[0]),
                maxcorr, str(bestdelay*3600), str(delay_index),
                signchange, corrthreshpass, dirthreshpass]

    datastore.append(dataline)
    logging.info("Maximum correlation value: " + str(maxval))
    logging.info("Minimum correlation value: " + str(minval))
    logging.info("The maximum correlation between " + causevar +
                 " and " + affectedvar + " is: " + str(maxcorr))
    # +1 needed due to the way delays defined in atx function
    logging.info("The corresponding delay is: " +
                 str(bestdelay*3600) +
                 " seconds, delay index number: " +
                 str(delay_index+1))
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
                maxval, str(bestdelay*3600), str(delay_index),
                threshpass]
    datastore.append(dataline)

    logging.info("The maximum TE between " + causevar +
                 " and " + affectedvar + " is: " + str(maxval))
    # +1 needed due to the way delays defined in atx function
    logging.info("The corresponding delay is delay index number: " +
                 str(delay_index+1))
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
    logging.basicConfig(level=logging.INFO)
    weight_array = np.empty((len(variables), len(variables)))
    delay_array = np.empty((len(variables), len(variables)))
    weight_array[:] = np.NAN
    delay_array[:] = np.NAN

    # Normalise inputdata to be safe
    inputdata = preprocessing.scale(inputdata, axis=0)

    if method == 'partial_correlation':

        threshcorr = (1.85*(size**(-0.41))) + (2.37*(size**(-0.53)))
        threshdir = 0.46*(size**(-0.16))

        logging.info("Directionbality threshold: " + str(threshdir))
        logging.info("Correlation threshold: " + str(threshcorr))

        data_header = ['causevar', 'affectedvar', 'base_corr',
                       'max_corr', 'max_delay', 'max_index',
                       'signchange', 'corrthreshpass', 'dirrthreshpass']

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

                    # This is similar to the vectorselection function in
                    # the transentropy module

                    causevardata = \
                        inputdata[:, causevarindex][-(size+delay+1):-(delay+1)]
                    print len(causevardata)
                    affectedvardata = \
                        inputdata[:, affectedvarindex][-(size+1):-1]
                    print len(affectedvardata)

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
                        del teCalc

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

#    if method == 'transfer_entropy':
#        shutdownJVM()

    return weight_array, delay_array, datastore, data_header
