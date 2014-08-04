"""This module calculates the gains (weights) of edges connecting
variables in the digraph.

Calculation of both Pearson's correlation and transfer entropy is supported.
Transfer entropy is calculated according to the global average of local
entropies method.
All weights are optimized with respect to time shifts between the time series
data vectors (i.e. cross-correlated).

The delay giving the maximum weight is returned, together with the maximum
weights.

All weights are tested for significance.
The Pearson's correlation weights are tested for signigicance according to
the parameters presented by Bauer2005.
The transfer entropy weights are tested for significance using a non-parametric
rank-order method using surrogate data generated according to the iterative
amplitude adjusted Fourier transform method (iAAFT).

@author: Simon Streicher, St. Elmo Wilken

"""
# Standard libraries
import os
import csv
import numpy as np
import h5py
import logging
import json

# Non-standard external libraries
import pygeonetwork
import jpype

# Own libraries
import config_setup
import transentropy
#import formatmatrices
import data_processing

#import datagen


class WeightcalcData:
    """Creates a data object from file and or function definitions for use in
    weight calculation methods.

    """
    def __init__(self, mode, case):
        # Get file locations from configuration file
        self.saveloc, self.casedir, infodynamicsloc = \
            config_setup.runsetup(mode, case)
        # Load case config file
        self.caseconfig = json.load(open(os.path.join(self.casedir, case +
                                    '_weightcalc' + '.json')))
        # Get data type
        self.datatype = self.caseconfig['datatype']
        # Get scenarios
        self.scenarios = self.caseconfig['scenarios']
        # Get methods
        self.methods = self.caseconfig['methods']

        # Start JVM if required
        if 'transfer_entropy' in self.methods:
            if not jpype.isJVMStarted():
                jpype.startJVM(jpype.getDefaultJVMPath(),
                               "-Xms32M",
                               "-Xmx512M",
                               "-ea",
                               "-Djava.class.path=" + infodynamicsloc)

        self.casename = case

    def scenariodata(self, scenario):
        """Retrieves data particular to each scenario for the case being
        investigated.

        """
        print "The scenario name is: " + scenario
        settings_name = self.caseconfig[scenario]['settings']
        connections_used = (self.caseconfig[settings_name]
                            ['use_connections'])
        bandgap_filtering = self.caseconfig[scenario]['bandgap_filtering']

        if self.datatype == 'file':
            # Get path to time series data input file in standard format
            # described in documentation under "Input data formats"
            raw_tsdata = os.path.join(self.casedir, 'data',
                                      self.caseconfig[scenario]['data'])

            # Retrieve connection matrix criteria from settings
            if connections_used:
                # Get connection (adjacency) matrix
                connectionloc = os.path.join(self.casedir, 'connections',
                                             self.caseconfig[scenario]
                                             ['connections'])
                self.connectionmatrix, _ = \
                    data_processing.read_connectionmatrix(connectionloc)

            # Get sampling rate and unit name
            self.sampling_rate = (self.caseconfig[settings_name]
                                  ['sampling_rate'])
            self.sampling_unit = (self.caseconfig[settings_name]
                                  ['sampling_unit'])
            # Get starting index
            self.startindex = self.caseconfig[settings_name]['startindex']
            # Convert timeseries data in CSV file to H5 data format
            datapath = data_processing.csv_to_h5(self.saveloc, raw_tsdata,
                                                 scenario, self.casename)
            # Read variables from orignal CSV file
            self.variables = data_processing.read_variables(raw_tsdata)
            # Get inputdata from H5 table created
            self.inputdata_raw = np.array(h5py.File(os.path.join(
                datapath, scenario + '.h5'), 'r')[scenario])

#        elif self.datatype == 'function':
#            raw_tsdata_gen = self.caseconfig[scenario]['datagen']
#            connectionloc = self.caseconfig[scenario]['connections']
#            # TODO: Store function arguments in scenario config file
#            samples = self.caseconfig['gensamples']
#            func_delay = self.caseconfig['delay']
#            # Get inputdata
#            self.inputdata_raw = eval('datagen.' + raw_tsdata_gen)(samples,
#                                                                   func_delay)
#            # Get the variables and connection matrix
#            self.connectionmatrix = eval('datagen.' + connectionloc)()
#            self.startindex = 0

        # Get delay type
        self.delaytype = self.caseconfig[settings_name]['delaytype']

        # Get size of sample vectors for tests
        # Must be smaller than number of samples generated
        self.testsize = self.caseconfig[settings_name]['testsize']
        # Get number of delays to test
        test_delays = self.caseconfig[scenario]['test_delays']
        # Define intervals of delays
        if self.delaytype == 'datapoints':
            # Include first n sampling intervals
            self.delays = range(test_delays + 1)
        elif self.delaytype == 'intervals':
            # Test delays at specified intervals
            self.delayinterval = \
                self.caseconfig[settings_name]['delay_interval']

            self.delays = [(val * self.delayinterval)
                           for val in range(test_delays + 1)]

        self.causevarindexes = self.caseconfig[scenario]['causevarindexes']
        if self.causevarindexes == 'all':
            self.causevarindexes = range(len(self.variables))
        self.affectedvarindexes = \
            self.caseconfig[scenario]['affectedvarindexes']
        if self.affectedvarindexes == 'all':
            self.affectedvarindexes = range(len(self.variables))

        # Normalise (mean centre and variance scale) the input data
        self.inputdata_normalised = \
            data_processing.normalise_data(raw_tsdata, self.inputdata_raw,
                                           self.saveloc, self.casename,
                                           scenario)
        if bandgap_filtering:
            low_freq = self.caseconfig[scenario]['low_freq']
            high_freq = self.caseconfig[scenario]['high_freq']
            self.inputdata_bandgapfiltered = \
                data_processing.bandgapfilter_data(raw_tsdata,
                                                   self.inputdata_normalised,
                                                   self.variables,
                                                   low_freq, high_freq,
                                                   self.saveloc, self.casename,
                                                   scenario)
            self.inputdata_originalrate = self.inputdata_bandgapfiltered
        else:
            self.inputdata_originalrate = self.inputdata_normalised

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
        elif self.delaytype == 'intervals':
            self.actual_delays = [int(round(delay/self.sampling_rate)) *
                                  self.sampling_rate for delay in self.delays]
            self.sample_delays = [int(round(delay/self.sampling_rate))
                                  for delay in self.delays]
        # Create descriptive dictionary for later use
        self.descriptions = data_processing.descriptive_dictionary(
            os.path.join(self.casedir, 'data', 'tag_descriptions.csv'))

        # FFT the data and write back in format that can be analysed in
        # TOPCAT in a plane plot
        data_processing.fft_calculation(raw_tsdata,
                                        self.inputdata_originalrate,
                                        self.variables,
                                        self.sampling_rate,
                                        self.sampling_unit,
                                        self.saveloc,
                                        self.casename,
                                        scenario)


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
        return [corrval]

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

        transent_directional = transent_fwd - transent_bwd
        transent_absolute = transent_fwd

        # Do not pass negatives on to weight array
#        if transent_directional < 0:
#            transent_directional = 0
#
#        if transent_absolute < 0:
#            transent_absolute = 0

        return [transent_directional, transent_absolute]

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

        # We already know that when dealing with transfer entropy
        # the weightlist will consist of a list of lists
        weightlist_directional = weightlist[0]
        weightlist_absolute = weightlist[1]

        size = weightcalcdata.testsize
        startindex = weightcalcdata.startindex

        # Do everything for the directional case

        maxval_directional = max(weightlist_directional)
        delay_index_directional = \
            weightlist_directional.index(maxval_directional)
        bestdelay_directional = \
            weightcalcdata.actual_delays[delay_index_directional]
        bestdelay_sample = \
            weightcalcdata.sample_delays[delay_index_directional]
        delay_array[affectedvarindex, causevarindex] = \
            bestdelay_directional

        logging.info("The maximum directional TE between " + causevar +
                     " and " + affectedvar + " is: " + str(maxval_directional))

        threshpass_directional = None
        if sigtest:
            # Calculate threshold for transfer entropy
            thresh_causevardata = \
                inputdata[:, causevarindex][startindex:startindex+size]
            thresh_affectedvardata = \
                inputdata[:, affectedvarindex][startindex + bestdelay_sample:
                                               startindex + size +
                                               bestdelay_sample]
            if te_thresh_method == 'rankorder':
                self.thresh_rankorder(thresh_affectedvardata.T,
                                      thresh_causevardata.T)

            elif te_thresh_method == 'sixsigma':
                self.thresh_sixsigma(thresh_affectedvardata.T,
                                     thresh_causevardata.T)

            logging.info("The directional TE threshold is: "
                         + str(self.threshent_directional))

            if maxval_directional >= self.threshent_directional:
                threshpass_directional = True
            else:
                threshpass_directional = False
                maxval_directional = 0

            logging.info("TE threshold passed: " + str(threshpass_directional))

        weight_array[affectedvarindex, causevarindex] = maxval_directional

        dataline = [causevar, affectedvar, str(weightlist_directional[0]),
                    maxval_directional, str(bestdelay_directional),
                    str(delay_index_directional),
                    threshpass_directional]

        datastore.append(dataline)

        logging.info("The corresponding delay is: "
                     + str(bestdelay_directional))
        logging.info("The TE with no delay is: " + str(weightlist[0][0]))

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

        surr_te_directional = \
            [surr_te_fwd[n] - surr_te_bwd[n] for n in range(num)]

        surr_te_absolute = [surr_te_fwd[n] for n in range(num)]

        return surr_te_directional, surr_te_absolute

    def thresh_rankorder(self, affected_data, causal_data):
        """Calculates the minimum threshold required for a transfer entropy
        value to be considered significant.

        Makes use of a 95% single-sided certainty and a rank-order method.
        This correlates to taking the maximum transfer entropy from 19
        surrogate transfer entropy calculations as the threshold,
        see Schreiber2000a.

        """
        surr_te_directional, surr_te_absolute = \
            self.calc_surr_te(affected_data, causal_data, 19)

        self.threshent_directional = max(surr_te_directional)
        self.threshent_absolute = max(surr_te_absolute)

    def thresh_sixsigma(self, affected_data, causal_data):
        """Calculates the minimum threshold required for a transfer entropy
        value to be considered significant.

        Makes use of a six sigma Gaussian check as done in Bauer2005 with 30
        samples of surrogate data.

        """
        surr_te_directional, surr_te_absolute = \
            self.calc_surr_te(affected_data, causal_data, 30)

        surr_te_directional_mean = np.mean(surr_te_directional)
        surr_te_directional_stdev = np.std(surr_te_directional)

        surr_te_absolute_mean = np.mean(surr_te_absolute)
        surr_te_absolute_stdev = np.std(surr_te_absolute)

        self.threshent_directional = (6 * surr_te_directional_stdev) + \
            surr_te_directional_mean

        self.threshent_absolute = (6 * surr_te_absolute_stdev) + \
            surr_te_absolute_mean


def estimate_delay(weightcalcdata, method, sigtest, scenario):
    """Determines the maximum weight between two variables by searching through
    a specified set of delays.

    method can be either 'cross_correlation' or 'transfer_entropy'

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

#    newconnectionmatrix = weightcalcdata.connectionmatrix
    # Substitute all rows and columns not used with zeros in connectionmatrix
#    for delindex in dellist:
#        newconnectionmatrix[:, delindex] = np.zeros(vardims)
#        newconnectionmatrix[delindex, :] = np.zeros(vardims)

    # Initiate headerline for weightstore file
    headerline = []
    # Create "Delay" as header for first row
    headerline.append('Delay')

    for affectedvarindex in weightcalcdata.affectedvarindexes:
            affectedvarname = weightcalcdata.variables[affectedvarindex]
            headerline.append(affectedvarname)

    # Store the weight calculation results in similar format as original data
    def writecsv_weightcalc(filename, items, header):
        """CSV writer customized for use in weightcalc function."""
        with open(filename, 'wb') as f:
            csv.writer(f).writerow(header)
            csv.writer(f).writerows(items)

    weightstoredir = config_setup.ensure_existance(
        os.path.join(weightcalcdata.saveloc, 'weightdata'), make=True)

    filename_template = os.path.join(weightstoredir, '{}_{}_{}_{}_{}.csv')

    for causevarindex in weightcalcdata.causevarindexes:
        causevar = weightcalcdata.variables[causevarindex]

        # Create filename for new CSV file containing weights between
        # this causevar and all the subsequent affectedvars
        def filename(name, method, causevar):
            return filename_template.format(weightcalcdata.casename,
                                            scenario, name, method, causevar)
        # Initiate datalines with delays
        datalines_directional = np.asarray(weightcalcdata.sample_delays)
        datalines_directional = datalines_directional[:, np.newaxis]
        datalines_absolute = datalines_directional.copy()

        for affectedvarindex in weightcalcdata.affectedvarindexes:
            affectedvar = weightcalcdata.variables[affectedvarindex]
            logging.info("Analysing effect of: " + causevar + " on " +
                         affectedvar)
#            if not(newconnectionmatrix[affectedvarindex,
#                                       causevarindex] == 0):
            weightlist = []
            directional_weightlist = []
            absolute_weightlist = []

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

                if len(weight) > 1:
                    # Iff weight contains directional as well as absolute
                    # weights, write to separate lists
                    directional_weightlist.append(weight[0])
                    absolute_weightlist.append(weight[1])
                else:
                    weightlist.append(weight[0])

            if len(weight) > 1:
                weightlist = [directional_weightlist, absolute_weightlist]

            # Combine weight data

                weights_thisvar_directional = np.asarray(weightlist[0])
                weights_thisvar_directional = \
                    weights_thisvar_directional[:, np.newaxis]

                weights_thisvar_absolute = np.asarray(weightlist[1])
                weights_thisvar_absolute = \
                    weights_thisvar_absolute[:, np.newaxis]

                datalines_directional = \
                    np.concatenate((datalines_directional,
                                    weights_thisvar_directional), axis=1)

                datalines_absolute = \
                    np.concatenate((datalines_absolute,
                                    weights_thisvar_absolute), axis=1)

                writecsv_weightcalc(filename('weights_directional', method,
                                             causevar),
                                    datalines_directional, headerline)

                writecsv_weightcalc(filename('weights_absolute', method,
                                             causevar),
                                    datalines_absolute, headerline)
            else:
                weights_thisvar_absolute = np.asarray(weightlist)
                weights_thisvar_absolute = \
                    weights_thisvar_absolute[:, np.newaxis]

                datalines_absolute = \
                    np.concatenate((datalines_absolute,
                                    weights_thisvar_absolute), axis=1)

                writecsv_weightcalc(filename('weights_absolute', method,
                                             causevar),
                                    datalines_absolute, headerline)

            [weight_array, delay_array, datastore] = \
                weightcalculator.report(weightcalcdata, causevarindex,
                                        affectedvarindex, weightlist,
                                        weight_array, delay_array,
                                        datastore, sigtest)

#        delays = np.asarray(weightcalcdata.sample_delays)
#        delays = delays[:, np.newaxis]
#        datalines = np.concatenate((delays, weights_allvars), axis=1)

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
                estimate_delay(weightcalcdata, method, sigtest, scenario)

            # Do noderanking immediately
#            looprank_static

            if writeoutput:
                # Define export directories and filenames
                weightdir = config_setup.ensure_existance(os.path.join(
                    weightcalcdata.saveloc, 'weightcalc'), make=True)
                filename_template = os.path.join(weightdir, '{}_{}_{}_{}.csv')

                def filename(name):
                    return filename_template.format(case, scenario,
                                                    method, name)

                # TODO
                # Convert arrays to lists in order to write mixed with strings
                # to CSV file

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
