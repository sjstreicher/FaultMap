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


def calc_max_partialcor(variables, connectionmatrix, inputdata,
                        dataset, sampling_rate, delays, size, delaytype):
    """Determines the maximum partial correlation between two variables.

    size refers to the number of elements of two time series data vectors used
    It is kept constant so as to eliminate any effect that different
    vector length might have on partial correlation

    inputdata should be normalised (mean centered and variance scaled)

    """
    logging.basicConfig(level=logging.INFO)
    max_val_array = np.empty((len(variables), len(variables)))
    max_delay_array = np.empty((len(variables), len(variables)))
    max_val_array[:] = np.NAN
    max_delay_array[:] = np.NAN

    data_header = ['causevar', 'affectedvar', 'base_corr',
                   'max_corr', 'max_delay',
                   'max_index', 'min_corr', 'min_delay', 'min_index',
                   'signchange']

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

                corrlist_abs = [abs(val) for val in corrlist]
                max_value_abs = max(corrlist_abs)
                min_value_abs = min(corrlist_abs)
                max_index = corrlist_abs.index(max_value_abs)
                max_value = corrlist[max_index]
                min_index = corrlist_abs.index(min_value_abs)
                min_value = corrlist[min_index]
                max_delay = actual_delays[max_index]
                min_delay = actual_delays[min_index]
                max_val_array[affectedvarindex, causevarindex] = max_value
                max_delay_array[affectedvarindex, causevarindex] = max_delay

                if (corrlist[0] / corrlist[max_index]) >= 0:
                    signchange = False
                else:
                    signchange = True

                dataline = [causevar, affectedvar, str(corrlist[0]),
                            max_value, str(max_delay*3600), str(max_index),
                            str(min_value), str(min_delay*3600),
                            str(min_index), signchange]

                datastore.append(dataline)

                logging.info("The maximum correlation between " + causevar
                             + " and " + affectedvar + " is at a delay of: " +
                             str(max_delay*3600) +
                             " seconds, delay index number: " + str(max_index))
                logging.info("The maximum correlation is: " + str(max_value) +
                             " while the minimum correlation is: " +
                             str(min_value) +
                             " which occurs at a delay of: " +
                             str(min_delay*3600) +
                             " seconds, delay index number: " + str(min_index))
                logging.info("The correlation with no delay is: "
                             + str(corrlist[0]))

            else:
                max_val_array[affectedvarindex, causevarindex] = np.NAN
                max_delay_array[affectedvarindex, causevarindex] = np.NAN

    return max_val_array, max_delay_array, datastore, data_header


def calc_transentropy_gainmatrix(connectionmatrix, tags_tsdata):
    """Calculates the local gains in terms of the transfer entropy between
    the variables.

    connectionmatrix is the adjacency matrix

    tags_tsdata contains the time series data for the tags with variables
    in colums and sampling instances in rows

    """

    #TODO: To be completed

    return None
