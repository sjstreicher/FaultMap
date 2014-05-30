"""This method is used to calculate the gains (weights) of edges connecting
variables in the network.

It allows for both the correlation and transfer entropy.
All weights are optimized with respect to time (i.e. cross-correlated)

The delay giving the maximum weight is returned, together with the maximum
weights.

All weights are tested for significance.
The following output options exits:

<<<To be completed>>>

@author: St. Elmo Wilken, Simon Streicher

"""
# Standard libraries
import os
import csv
import numpy as np
import h5py
import logging
import json
import sklearn
import sklearn.preprocessing

# Less standard libraries
import pygeonetwork
import jpype

# Own libraries
import config_setup
import transentropy
import formatmatrices

import datagen


class WeightcalcData:
    """Creates a data object from file and or function definitions for use in
    weight calculation methods.

    """
    def __init__(self, mode, case):
        # Get locations from configuration file
        self.saveloc, self.casedir, infodynamicsloc = \
            config_setup.runsetup(mode, case)
        # Load case config file
        self.caseconfig = json.load(open(os.path.join(self.casedir, case +
                                    '_weightcalc' + '.json')))
        # Get scenarios
        self.scenarios = self.caseconfig['scenarios']
        # Get sampling rate
        self.sampling_rate = self.caseconfig['sampling_rate']
        # Get data type
        self.datatype = self.caseconfig['datatype']
        # Get delay type
        self.delaytype = self.caseconfig['delaytype']
        # Get methods
        self.methods = self.caseconfig['methods']
        # Get size of sample vectors for tests
        # Must be smaller than number of samples generated
        self.testsize = self.caseconfig['testsize']
        # Get number of delays to test
        test_delays = self.caseconfig['test_delays']

        if self.delaytype == 'datapoints':
        # Include first n sampling intervals
            self.delays = range(test_delays + 1)
        elif self.delaytype == 'timevalues':
        # Include first n 10-second shifts
            self.delays = [val * (10.0/3600.0) for val in
                           range(test_delays + 1)]

        # Start JVM if required
        if 'transfer_entropy' in self.methods:
            if not jpype.isJVMStarted():
                jpype.startJVM(jpype.getDefaultJVMPath(),
                               "-Xms32M",
                               "-Xmx512M",
                               "-ea",
                               "-Djava.class.path=" + infodynamicsloc)

    def scenariodata(self, scenario):
        """Retrieves data particular to each scenario for the case being
        investigated.

        """
        # Get time series data
        if self.datatype == 'file':
            # Get time series data
            tags_tsdata = os.path.join(self.casedir, 'data',
                                       self.caseconfig[scenario]['data'])
            # Get connection (adjacency) matrix
            connectionloc = os.path.join(self.casedir, 'connections',
                                         self.caseconfig[scenario]
                                         ['connections'])
            # Get dataset name
            dataset = self.caseconfig[scenario]['dataset']
            # Get starting index
            self.startindex = self.caseconfig['startindex']
            # Get inputdata
            self.inputdata_raw = np.array(h5py.File(tags_tsdata, 'r')[dataset])
            # Get the variables and connection matrix
            [self.variables, self.connectionmatrix] = \
                formatmatrices.read_connectionmatrix(connectionloc)

        elif self.datatype == 'function':
            tags_tsdata_gen = self.caseconfig[scenario]['datagen']
            connectionloc = self.caseconfig[scenario]['connections']
            # TODO: Store function arguments in scenario config file
            samples = self.caseconfig['gensamples']
            func_delay = self.caseconfig['delay']
            # Get inputdata
            self.inputdata_raw = eval('datagen.' + tags_tsdata_gen)(samples, func_delay)
            # Get the variables and connection matrix
            [self.variables, self.connectionmatrix] = eval('datagen.' + connectionloc)()
            self.startindex = 0

        self.causevarindexes = self.caseconfig[scenario]['causevarindexes']
        if self.causevarindexes == 'all':
            self.causevarindexes = range(len(self.variables))
        self.affectedvarindexes = \
            self.caseconfig[scenario]['affectedvarindexes']
        if self.affectedvarindexes == 'all':
            self.affectedvarindexes = range(len(self.variables))

        # Normalise (mean centre and variance scale) the input data
        self.inputdata_originalrate = \
            sklearn.preprocessing.scale(self.inputdata_raw,
                                        axis=0)

        # Subsample data if required
        # Get sub_sampling interval
        sub_sampling_interval = \
            self.caseconfig[scenario]['sub_sampling_interval']
        # TOOD: Use proper pandas.tseries.resample techniques
            # if it will really add any functionality
        self.inputdata = self.inputdata_originalrate[0::sub_sampling_interval]

        if self.delaytype == 'datapoints':
                self.actual_delays = [(delay * self.sampling_rate *
                                       sub_sampling_interval)
                                      for delay in self.delays]
                self.sample_delays = self.delays
        elif self.delaytype == 'timevalues':
            self.actual_delays = [int(round(delay/self.sampling_rate)) *
                                  self.sampling_rate for delay in self.delays]
            self.sample_delays = [int(round(delay/self.sampling_rate))
                                  for delay in self.delays]


class CorrWeightcalc:
    """This class provides methods for calculating the weights according to the
    cross-correlation method.

    """
    def __init__(self, weightcalcdata):
        """Read the files or functions and returns required data fields.

        """
        self.threshcorr = (1.85*(weightcalcdata.testsize**(-0.41))) + \
            (2.37*(weightcalcdata.testsize**(-0.53)))
        self.threshdir = 0.46*(weightcalcdata.testsize**(-0.16))
#        logging.info("Directionality threshold: " + str(self.threshdir))
#        logging.info("Correlation threshold: " + str(self.threshcorr))

        self.data_header = ['causevar', 'affectedvar', 'base_corr',
                            'max_corr', 'max_delay', 'max_index',
                            'signchange', 'corrthreshpass',
                            'dirrthreshpass', 'dirval']

    def calcweight(self, causevardata, affectedvardata):
        """Calculates the correlation between two vectors containing
        timer series data.

        """
        corrval = np.corrcoef(causevardata.T, affectedvardata.T)[1, 0]
        return corrval

    def report(self, weightcalcdata, causevarindex, affectedvarindex,
               weightlist, weight_array, delay_array, datastore, sigtest):
        """Calculates and reports the relevant output for each combination
        of variables tested.

        """
        variables = weightcalcdata.variables
        causevar = variables[causevarindex]
        affectedvar = variables[affectedvarindex]

        maxval = max(weightlist)
        minval = min(weightlist)
        # Value used to break tie between maxval and minval if 1 and -1
        tol = 0.
        # Always select maxval if both are equal
        # This is to ensure that 1 is selected above -1
        if (maxval + (minval + tol)) >= 0:
            maxcorr = maxval
        else:
            maxcorr = minval

        delay_index = weightlist.index(maxcorr)

        # Correlation thresholds from Bauer2008 eq. 4
        maxcorr_abs = max(maxval, abs(minval))
        bestdelay = weightcalcdata.actual_delays[delay_index]
        directionindex = 2 * (abs(maxval + minval) /
                              (maxval + abs(minval)))

        signchange = not ((weightlist[0] / weightlist[delay_index]) >= 0)
        corrthreshpass = (maxcorr_abs >= self.threshcorr)
        dirthreshpass = (directionindex >= self.threshdir)

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

        corrthreshpass = None
        dirthreshpass = None
        if sigtest:
            corrthreshpass = (maxcorr_abs >= self.threshcorr)
            dirthreshpass = (directionindex >= self.threshdir)
            if not (corrthreshpass and dirthreshpass):
                maxcorr = 0

        weight_array[affectedvarindex, causevarindex] = maxcorr

#        # Replace all nan by zero
#        nanlocs = np.isnan(weight_array)
#        weight_array[nanlocs] = 0

        delay_array[affectedvarindex, causevarindex] = bestdelay

        dataline = [causevar, affectedvar, str(weightlist[0]),
                    maxcorr, str(bestdelay), str(delay_index),
                    signchange, corrthreshpass, dirthreshpass, directionindex]

        datastore.append(dataline)

        return weight_array, delay_array, datastore


class TransentWeightcalc:
    """This class provides methods for calculating the weights according to
    the transfer entropy method.

    """

    def __init__(self, weightcalcdata):
        self.data_header = ['causevar', 'affectedvar', 'base_ent',
                            'max_ent', 'max_delay', 'max_index', 'threshpass']
        # Setup Java class for infodynamics toolkit
        self.teCalc = transentropy.setup_infodynamics_te()

    def calcweight(self, causevardata, affectedvardata):
        """"Calculates the transfer entropy between two vectors containing
        timer series data.

        """
        # Calculate transfer entropy as the difference
        # between the forward and backwards entropy
        transent_fwd = \
            transentropy.calc_infodynamics_te(self.teCalc, affectedvardata.T,
                                              causevardata.T)
        transent_bwd = \
            transentropy.calc_infodynamics_te(self.teCalc, causevardata.T,
                                              affectedvardata.T)
        transent = transent_fwd - transent_bwd

        # Do not pass negatives on to weight array
        if transent < 0:
            transent = 0

        return transent
        # TODO: Offer option for raw transfer entopy values as well

    def report(self, weightcalcdata, causevarindex, affectedvarindex,
               weightlist, weight_array, delay_array, datastore, sigtest,
               te_thresh_method='rankorder'):
        """Calculates and reports the relevant output for each combination
        of variables tested.

        """

        variables = weightcalcdata.variables
        causevar = variables[causevarindex]
        affectedvar = variables[affectedvarindex]
        inputdata = weightcalcdata.inputdata

        maxval = max(weightlist)
        delay_index = weightlist.index(maxval)
        bestdelay = weightcalcdata.actual_delays[delay_index]
        bestdelay_sample = weightcalcdata.sample_delays[delay_index]
        delay_array[affectedvarindex, causevarindex] = bestdelay

        size = weightcalcdata.testsize
        startindex = weightcalcdata.startindex

        logging.info("The maximum TE between " + causevar +
                     " and " + affectedvar + " is: " + str(maxval))

        threshpass = None
        if sigtest:
            # Calculate threshold for transfer entropy
            thresh_causevardata = \
                inputdata[:, causevarindex][startindex:startindex+size]
            thresh_affectedvardata = \
                inputdata[:, affectedvarindex][startindex+bestdelay_sample:
                                               startindex+size+bestdelay_sample]
            if te_thresh_method == 'rankorder':
                self.thresh_rankorder(thresh_affectedvardata.T,
                                      thresh_causevardata.T)

            elif te_thresh_method == 'sixsigma':
                self.thresh_sixsigma(thresh_affectedvardata.T,
                                     thresh_causevardata.T)

            logging.info("The TE threshold is: " + str(self.threshent))

            if maxval >= self.threshent:
                threshpass = True
            else:
                threshpass = False
                maxval = 0

            logging.info("TE threshold passed: " + str(threshpass))

        weight_array[affectedvarindex, causevarindex] = maxval

        dataline = [causevar, affectedvar, str(weightlist[0]),
                    maxval, str(bestdelay), str(delay_index),
                    threshpass]
        datastore.append(dataline)

        logging.info("The corresponding delay is: " + str(bestdelay))
        logging.info("The TE with no delay is: " + str(weightlist[0]))

        return weight_array, delay_array, datastore

    def calc_surr_te(self, affected_data, causal_data, num):
        """Calculates surrogate transfer entropy values for significance
        threshold purposes.

        Generates surrogate time series data by making use of the iAAFT method
        (see Schreiber 2000a).

        Returns list of surrogate transfer entropy values of length num.

        """
        tsdata = np.array([affected_data, causal_data])
        # TODO: Research required number of iterations for good surrogate data
        surr_tsdata = \
            [pygeonetwork.surrogates.Surrogates.SmallTestData().
                get_refined_AAFT_surrogates(tsdata, 10)
             for n in range(num)]

        surr_te_fwd = [transentropy.calc_infodynamics_te(self.teCalc,
                                                         surr_tsdata[n][0],
                                                         surr_tsdata[n][1])
                       for n in range(num)]
        surr_te_bwd = [transentropy.calc_infodynamics_te(self.teCalc,
                                                         surr_tsdata[n][1],
                                                         surr_tsdata[n][0])
                       for n in range(num)]

        surr_te = [surr_te_fwd[n] - surr_te_bwd[n] for n in range(num)]

        return surr_te

    def thresh_rankorder(self, affected_data, causal_data):
        """Calculates the minimum threshold required for a transfer entropy
        value to be considered significant.

        Makes use of a 95% single-sided certainty and a rank-order method.
        This correlates to taking the maximum transfer entropy from 19
        surrogate transfer entropy calculations as the threshold,
        see Schreiber2000a.

        """
        surrte = self.calc_surr_te(affected_data, causal_data, 19)

        self.threshent = max(surrte)

    def thresh_sixsigma(self, affected_data, causal_data):
        """Calculates the minimum threshold required for a transfer entropy
        value to be considered significant.

        Makes use of a six sigma Gaussian check as done in Bauer2005 with 30
        samples of surrogate data.

        """
        surrte = self.calc_surr_te(affected_data, causal_data, 30)

        surrte_mean = np.mean(surrte)
        surrte_stdev = np.std(surrte)

        self.threshent = (6 * surrte_stdev) + surrte_mean


def estimate_delay(weightcalcdata, method, sigtest):
    """Determines the maximum weight between two variables by searching through
    a specified set of delays.

    method can be either 'pearson_correlation' or 'transfer_entropy'

    """
    if method == 'cross_correlation':
        weightcalculator = CorrWeightcalc(weightcalcdata)
    elif method == 'transfer_entropy':
        weightcalculator = TransentWeightcalc(weightcalcdata)
    elif method == 'partial_correlation':
        weightcalculator = PartialCorrWeightcalc(weightcalcdata)

    startindex = weightcalcdata.startindex
    size = weightcalcdata.testsize
    data_header = weightcalculator.data_header
    vardims = len(weightcalcdata.variables)
    weight_array = np.empty((vardims, vardims))
    delay_array = np.empty((vardims, vardims))
    weight_array[:] = np.NAN
    delay_array[:] = np.NAN
    datastore = []

    dellist = []
    for index in range(vardims):
        if index not in weightcalcdata.causevarindexes:
            dellist.append(index)
            logging.info("Deleted column " + str(index))

    newconnectionmatrix = weightcalcdata.connectionmatrix
    # Substitute all rows and columns not used with zeros in connectionmatrix
    for delindex in dellist:
        newconnectionmatrix[:, delindex] = np.zeros(vardims)
        newconnectionmatrix[delindex, :] = np.zeros(vardims)

    for causevarindex in weightcalcdata.causevarindexes:
        causevar = weightcalcdata.variables[causevarindex]
        for affectedvarindex in weightcalcdata.affectedvarindexes:
            affectedvar = weightcalcdata.variables[affectedvarindex]
            logging.info("Analysing effect of: " + causevar + " on " +
                         affectedvar)
            if not(newconnectionmatrix[affectedvarindex,
                                       causevarindex] == 0):
                weightlist = []
                for delay in weightcalcdata.sample_delays:
                    logging.info("Now testing delay: " + str(delay))

                    causevardata = \
                        (weightcalcdata.inputdata[:, causevarindex]
                            [startindex:startindex+size])

                    affectedvardata = \
                        (weightcalcdata.inputdata[:, affectedvarindex]
                            [startindex+delay:startindex+size+delay])

                    weight = weightcalculator.calcweight(causevardata,
                                                         affectedvardata)
                    weightlist.append(weight)

                [weight_array, delay_array, datastore] = \
                    weightcalculator.report(weightcalcdata, causevarindex,
                                            affectedvarindex, weightlist,
                                            weight_array, delay_array,
                                            datastore, sigtest)

    # Delete entries from weightcalc matrix not used
    # Delete all rows and columns listed in dellist
    # from weight_array
    weight_array = np.delete(weight_array, dellist, 1)
    weight_array = np.delete(weight_array, dellist, 0)

    # Do the same for delay_array
    delay_array = np.delete(delay_array, dellist, 1)
    delay_array = np.delete(delay_array, dellist, 0)

    return weight_array, delay_array, datastore, data_header


def writecsv_weightcalc(filename, items, header):
    """CSV writer customized for use in weightcalc function."""

    with open(filename, 'wb') as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(items)


def weightcalc(mode, case, sigtest, writeoutput):
    """Reports the maximum weight as well as associated delay
    obtained by shifting the affected variable behind the causal variable a
    specified set of delays.

    Supports calculating weights according to either correlation or transfer
    entropy metrics.

    """

    weightcalcdata = WeightcalcData(mode, case)

    for scenario in weightcalcdata.scenarios:
        logging.info("Running scenario {}".format(scenario))
        # Update scenario-specific fields of weightcalcdata object
        weightcalcdata.scenariodata(scenario)
        for method in weightcalcdata.methods:
            logging.info("Method: " + method)

            # TODO: Get data_header directly
            [weight_array, delay_array, datastore, data_header] = \
                estimate_delay(weightcalcdata, method, sigtest)

            if writeoutput:
                # Define export directories and filenames
                weightdir = config_setup.ensure_existance(os.path.join(
                    weightcalcdata.saveloc, 'weightcalc'), make=True)
                filename_template = os.path.join(weightdir, '{}_{}_{}_{}.csv')

                def filename(name):
                    return filename_template.format(case, scenario,
                                                    method, name)
                # Write arrays to file
                np.savetxt(filename('maxweight_array'), weight_array,
                           delimiter=',')
                np.savetxt(filename('delay_array'), delay_array, delimiter=',')
                # Write datastore to file
                writecsv_weightcalc(filename('weightcalc_data'), datastore,
                                    data_header)


class PartialCorrWeightcalc:
    """This class provides methods for calculating the weights according to
    the transfer entropy method.

    """

    def __init__(self, weightcalcdata):
        return None

    def partialcorr_gainmatrix(self, weightcalcdata):
        """Calculates the local gains in terms of the partial (Pearson's)
        correlation between the variables.

        connectionmatrix is the adjacency matrix

        tags_tsdata contains the time series data for the tags with variables
        in colums and sampling instances in rows

        """
        startindex = weightcalcdata.startindex
        size = weightcalcdata.testsize
        vardims = len(weightcalcdata.variables)

        # Get inputdata and initial connectionmatrix
        calcdata = (weightcalcdata.inputdata[:, :]
                    [startindex:startindex+size])
        newconnectionmatrix = weightcalcdata.connectionmatrix
        newvariables = weightcalcdata.variables

        # Delete all variables from data matrix whose standard deviation
        # is zero.
        # This is not perfectly robust.
#        dellist = []
#        for col in range(calcdata.shape[1]):
#            stdev = np.std(calcdata[:, col])
#            if stdev == 0:
#                dellist.append(col)
#                logging.info("Will delete column " + str(col))

        # Delete all columns not listed in causevarindexes
        dellist = []
        for index in range(vardims):
            if index not in weightcalcdata.causevarindexes:
                dellist.append(index)
                logging.info("Deleted column " + str(index))

        # Delete all columns listed in dellist from calcdata
        newcalcdata = np.delete(calcdata, dellist, 1)

        # Delete all indexes listed in dellist from variables
        newvariables = np.delete(newvariables, dellist)

        # Delete all rows and columns listed in dellist
        # from connectionmatrix
        newconnectionmatrix = np.delete(newconnectionmatrix, dellist, 1)
        newconnectionmatrix = np.delete(newconnectionmatrix, dellist, 0)

        # Calculate correlation matrix
        correlationmatrix = np.corrcoef(newcalcdata.T)
        # Calculate partial correlation matrix
        p_matrix = np.linalg.inv(correlationmatrix)
        d = p_matrix.diagonal()
        partialcorrelationmatrix = \
            np.where(newconnectionmatrix,
                     -p_matrix/np.abs(np.sqrt(np.outer(d, d))), 0)

        return partialcorrelationmatrix, newconnectionmatrix, newvariables


# TODO: This function is a clone of the object method above
# and therefore redundant but used in the transient ranking algorithm.
# It will be incorporated as soon as it is high enough priority
def calc_partialcorr_gainmatrix(connectionmatrix, tags_tsdata, *dataset):
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


def partialcorrcalc(mode, case, writeoutput):
    """Returns the partial correlation matrix.

    Does not support optimizing with respect to time delays.

    """
    weightcalcdata = WeightcalcData(mode, case)
    partialmatcalculator = PartialCorrWeightcalc(weightcalcdata)

    for scenario in weightcalcdata.scenarios:
        logging.info("Running scenario {}".format(scenario))
        # Update scenario-specific fields of weightcalcdata object
        weightcalcdata.scenariodata(scenario)

        partialcorrmat, connectionmatrix, variables = partialmatcalculator.\
            partialcorr_gainmatrix(weightcalcdata)

        if writeoutput:
             # Define export directories and filenames
            partialmatdir = config_setup.ensure_existance(os.path.join(
                weightcalcdata.saveloc, 'partialcorr'), make=True)
            filename_template = os.path.join(partialmatdir, '{}_{}_{}.csv')

            def filename(name):
                return filename_template.format(case, scenario, name)
            # Write arrays to file
            np.savetxt(filename('partialcorr_array'), partialcorrmat,
                       delimiter=',')

            np.savetxt(filename('connectionmatrix'), connectionmatrix,
                       delimiter=',')

            writecsv_weightcalc(filename('variables'), variables,
                                variables)
