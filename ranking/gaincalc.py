"""This method is imported by formatmatrices

@author: St. Elmo Wilken, Simon Streicher

"""

import csv
import numpy as np
import h5py
import logging


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


def calc_max_partialcorr_delay(variables, connectionmatrix, inputdata,
                               dataset, sampling_rate, delays,
                               size, delaytype):
    """Determines the maximum partial correlation between two variables.

    size refers to the number of elements of two time series data vectors used
    It is kept constant so as to eliminate any effect that different
    vector length might have on partial correlation

    inputdata should be normalised (mean centered and variance scaled)

    """
    logging.basicConfig(level=logging.INFO)
    corr_array = np.empty((len(variables), len(variables)))
    delay_array = np.empty((len(variables), len(variables)))
    corr_array[:] = np.NAN
    delay_array[:] = np.NAN

    data_header = ['causevar', 'affectedvar', 'base_corr',
                   'max_corr', 'max_delay', 'max_index',
                   'signchange', 'corrthreshpass', 'dirrthreshpass']

    threshcorr = (1.85*(size**(-0.41))) + (2.37*(size**(-0.53)))
    threshdir = 0.46*(size**(-0.16))

    datastore = []
    if delaytype == 'datapoints':
        actual_delays = [delay * sampling_rate for delay in delays]
    elif delaytype == 'timevalues':
        actual_delays = [int(round(delay/sampling_rate)) * sampling_rate
                         for delay in delays]
    for causevarindex, causevar in enumerate(variables):
        logging.info("Analysing effect of: " + causevar)
        for affectedvarindex, affectedvar in enumerate(variables):
            if not(connectionmatrix[affectedvarindex, causevarindex] == 0):
                corrlist = []
                for delay in delays:
                    if delaytype == 'datapoints':
                        sample_delay = delay
                    else:
                        sample_delay = int(round(delay/sampling_rate))
                    causevardata = \
                        inputdata[:, causevarindex][-(size+sample_delay):
                                                    -1-sample_delay]
                    affectedvardata = \
                        inputdata[:, affectedvarindex][sample_delay+1:
                                                       sample_delay+size]
                    corrval = \
                        np.corrcoef(causevardata.T, affectedvardata.T)[1, 0]
                    corrlist.append(corrval)

                maxval = max(corrlist)
                minval = min(corrlist)
                if (maxval + minval) >= 0:
                    delay_index = corrlist.index(maxval)
                    maxcorr = maxval
                else:
                    delay_index = corrlist.index(minval)
                    maxcorr = minval
                # Bauer2008 eq. 4
                maxcorr_abs = max(maxval, abs(minval))
                bestdelay = actual_delays[delay_index]
                directionindex = 2 * (abs(maxval + minval) /
                                      (maxval + abs(minval)))
                corr_array[affectedvarindex, causevarindex] = maxcorr
                delay_array[affectedvarindex, causevarindex] = bestdelay

                if (corrlist[0] / corrlist[delay_index]) >= 0:
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

                dataline = [causevar, affectedvar, str(corrlist[0]),
                            maxcorr, str(bestdelay*3600), str(delay_index),
                            signchange, corrthreshpass, dirthreshpass]

                datastore.append(dataline)
                logging.info("Maximum correlation value: " + str(maxval))
                logging.info("Minimum correlation value: " + str(minval))
                logging.info("The maximum correlation between " + causevar +
                             " and " + affectedvar + " is: " + str(maxcorr))
                logging.info("The corresponding delay is: " +
                             str(bestdelay*3600) +
                             " seconds, delay index number: " +
                             str(delay_index))
                logging.info("The correlation with no delay is: "
                             + str(corrlist[0]))
#                logging.info("Correlation threshold: " + str(threshcorr))
                logging.info("Correlation threshold passed: " +
                             str(corrthreshpass))
#                logging.info("Directionbality threshold: " + str(threshdir))
                logging.info("Directionality value: " + str(directionindex))
                logging.info("Directionality threshold passed: " +
                             str(dirthreshpass))

            else:
                corr_array[affectedvarindex, causevarindex] = np.NAN
                delay_array[affectedvarindex, causevarindex] = np.NAN

    return corr_array, delay_array, datastore, data_header


def calc_transentropy_gainmatrix(connectionmatrix, tags_tsdata):
    """Calculates the local gains in terms of the transfer entropy between
    the variables.

    connectionmatrix is the adjacency matrix

    tags_tsdata contains the time series data for the tags with variables
    in colums and sampling instances in rows

    """

    #TODO: To be completed

    return None
