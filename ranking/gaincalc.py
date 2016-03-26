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
import time
import sklearn.preprocessing

# Own libraries
import config_setup
import data_processing

# Gain calculators
from gaincalculators import (PartialCorrWeightcalc, CorrWeightcalc,
                             TransentWeightcalc)

import gaincalc_oneset
import multiprocessing


class WeightcalcData:
    """Creates a data object from file and or function definitions for use in
    weight calculation methods.

    """
    def __init__(self, mode, case, single_entropies, fftcalc,
                 do_multiprocessing):
        # Get file locations from configuration file
        self.saveloc, self.caseconfigdir, \
            self.casedir, self.infodynamicsloc = \
            config_setup.runsetup(mode, case)
        # Load case config file
        self.caseconfig = json.load(
            open(os.path.join(self.caseconfigdir, case +
                              '_weightcalc' + '.json')))
        # Get data type
        self.datatype = self.caseconfig['datatype']
        # Get scenarios
        self.scenarios = self.caseconfig['scenarios']
        # Get methods
        self.methods = self.caseconfig['methods']

        self.do_multiprocessing = do_multiprocessing

        self.casename = case

        # Flag for calculating single signal entropies
        self.single_entropies = single_entropies
        # Flag for calculating FFT of all signals
        self.fftcalc = fftcalc

    def scenariodata(self, scenario):
        """Retrieves data particular to each scenario for the case being
        investigated.

        """
        print "The scenario name is: " + scenario

        self.settings_set = self.caseconfig[scenario]['settings']

    def setsettings(self, scenario, settings_name):

        self.connections_used = (self.caseconfig[settings_name]
                                 ['use_connections'])
        self.transient = self.caseconfig[settings_name]['transient']
        self.normalize = self.caseconfig[settings_name]['normalize']
        self.sigtest = self.caseconfig[settings_name]['sigtest']
        if self.sigtest:
            # The transfer entropy threshold calculation method be either
            # 'sixsigma' or 'rankorder'
            self.te_thresh_method = \
                self.caseconfig[settings_name]['te_thresh_method']
            # The transfer entropy surrogate generation method be either
            # 'iAAFT' or 'random_shuffle'
            self.te_surr_method = \
                self.caseconfig[settings_name]['te_surr_method']
        self.allthresh = self.caseconfig[settings_name]['allthresh']

        # Get sampling rate and unit name
        self.sampling_rate = (self.caseconfig[settings_name]
                              ['sampling_rate'])
        self.sampling_unit = (self.caseconfig[settings_name]
                              ['sampling_unit'])
        # Get starting index
        self.startindex = self.caseconfig[settings_name]['startindex']

        # Get any parameters for Kraskov method
        if 'transfer_entropy_kraskov' in self.methods:
            self.additional_parameters = \
                self.caseconfig[settings_name]['additional_parameters']

        if self.datatype == 'file':
            # Get path to time series data input file in standard format
            # described in documentation under "Input data formats"
            raw_tsdata = os.path.join(self.casedir, 'data',
                                      self.caseconfig[scenario]['data'])

            # Retrieve connection matrix criteria from settings
            if self.connections_used:
                # Get connection (adjacency) matrix
                connectionloc = os.path.join(self.casedir, 'connections',
                                             self.caseconfig[scenario]
                                             ['connections'])
                self.connectionmatrix, _ = \
                    data_processing.read_connectionmatrix(connectionloc)

            # Convert timeseries data in CSV file to H5 data format
            datapath = data_processing.csv_to_h5(self.saveloc, raw_tsdata,
                                                 scenario, self.casename)
            # Read variables from orignal CSV file
            self.variables = data_processing.read_variables(raw_tsdata)
            # Get inputdata from H5 table created
            self.inputdata_raw = np.array(h5py.File(os.path.join(
                datapath, scenario + '.h5'), 'r')[scenario])

            if self.normalize:
                # Normalise (mean centre and variance scale) the input data
                self.inputdata_normstep = \
                    data_processing.normalise_data(raw_tsdata,
                                                   self.inputdata_raw,
                                                   self.saveloc, self.casename,
                                                   scenario)
            else:
                # Still norm centre the data
                # This breaks when trying to use discrete methods
#                self.inputdata_normstep = data_processing.subtract_mean(
#                    self.inputdata_raw)
                self.inputdata_normstep = self.inputdata_raw

        elif self.datatype == 'function':
            import datagen
            raw_tsdata_gen = self.caseconfig[scenario]['datagen']
            if self.connections_used:
                connectionloc = self.caseconfig[scenario]['connections']
            # TODO: Store function arguments in scenario config file
            params = self.caseconfig[settings_name]['datagen_params']
            # Get inputdata
            self.inputdata_raw = \
                eval('datagen.' + raw_tsdata_gen)(params)
            self.inputdata_raw = np.asarray(self.inputdata_raw)
            # Get the variables and connection matrix
            self.variables, self.connectionmatrix = \
                eval('datagen.' + connectionloc)()

            if self.normalize:
                self.inputdata_normstep = \
                    sklearn.preprocessing.scale(self.inputdata_raw, axis=0)
            else:
                self.inputdata_normstep = self.inputdata_raw

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

        bandgap_filtering = self.caseconfig[scenario]['bandgap_filtering']
        if bandgap_filtering:
            low_freq = self.caseconfig[scenario]['low_freq']
            high_freq = self.caseconfig[scenario]['high_freq']
            self.inputdata_bandgapfiltered = \
                data_processing.bandgapfilter_data(raw_tsdata,
                                                   self.inputdata_normstep,
                                                   self.variables,
                                                   low_freq, high_freq,
                                                   self.saveloc, self.casename,
                                                   scenario)
            self.inputdata_originalrate = self.inputdata_bandgapfiltered
        else:
            self.inputdata_originalrate = self.inputdata_normstep

        # Subsample data if required
        # Get sub_sampling interval
        self.sub_sampling_interval = \
            self.caseconfig[settings_name]['sub_sampling_interval']
        # TODO: Use proper pandas.tseries.resample techniques
        # if it will really add any functionality
        # TODO: Investigate use of forward-backward Kalman filters
        self.inputdata = \
            self.inputdata_originalrate[0::self.sub_sampling_interval]

        if self.transient:
            self.boxnum = self.caseconfig[settings_name]['boxnum']
            self.boxsize = self.caseconfig[settings_name]['boxsize']
        else:
            self.boxnum = 1  # Only a single box will be used
            self.boxsize = self.inputdata.shape[0] * \
                self.sampling_rate
            # This box should now return the same size
            # as the original data file - but it does not play a role at all
            # in the actual box determination for the case of boxnum = 1

        # Select which of the boxes to evaluate
        if self.transient:
            self.boxindexes = self.caseconfig[scenario]['boxindexes']
            if self.boxindexes == 'all':
                self.boxindexes = range(self.boxnum)
        else:
            self.boxindexes = [0]

        if self.delaytype == 'datapoints':
            self.actual_delays = [(delay * self.sampling_rate *
                                  self.sub_sampling_interval)
                                  for delay in self.delays]
            self.sample_delays = self.delays
        elif self.delaytype == 'intervals':
            self.actual_delays = [int(round(delay/self.sampling_rate)) *
                                  self.sampling_rate for delay in self.delays]
            self.sample_delays = [int(round(delay/self.sampling_rate))
                                  for delay in self.delays]

        # Create descriptive dictionary for later use
        # This will need to be approached slightly differently to allow for
        # different formats under the same "plant"
#        self.descriptions = data_processing.descriptive_dictionary(
#            os.path.join(self.casedir, 'data', 'tag_descriptions.csv'))

        # FFT the data and write back in format that can be analysed in
        # TOPCAT in a plane plot
        if self.fftcalc:
            data_processing.fft_calculation(raw_tsdata,
                                            self.inputdata_originalrate,
                                            self.variables,
                                            self.sampling_rate,
                                            self.sampling_unit,
                                            self.saveloc,
                                            self.casename,
                                            scenario)


def writecsv_weightcalc(filename, items, header):
    """CSV writer customized for use in weightcalc function."""

    with open(filename, 'wb') as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(items)


def calc_weights(weightcalcdata, method, scenario):
    """Determines the maximum weight between two variables by searching through
    a specified set of delays.

    method can be one of the following:
    'cross_correlation'
    'partial_correlation' -- does not support time delays
    'transfer_entropy_kernel'
    'transfer_entropy_kraskov'

    """
    # TODO: Allow for calculation of significance values at each data point
    # and storing in files similar to weight calculations

    if method == 'cross_correlation':
        weightcalculator = CorrWeightcalc(weightcalcdata)
    elif method == 'transfer_entropy_kernel':
        weightcalculator = TransentWeightcalc(weightcalcdata, 'kernel')
    elif method == 'transfer_entropy_kraskov':
        weightcalculator = TransentWeightcalc(weightcalcdata, 'kraskov')
    elif method == 'transfer_entropy_discrete':
        weightcalculator = TransentWeightcalc(weightcalcdata, 'discrete')
    elif method == 'partial_correlation':
        weightcalculator = PartialCorrWeightcalc(weightcalcdata)

    if weightcalcdata.sigtest:
        sigstatus = 'sigtested'
    elif not weightcalcdata.sigtest:
        sigstatus = 'nosigtest'

    if method == 'transfer_entropy_kraskov':
        if weightcalcdata.additional_parameters['auto_embed']:
            embedstatus = 'autoembedding'
        else:
            embedstatus = 'naive'
    else:
        embedstatus = 'naive'

    vardims = len(weightcalcdata.variables)
    startindex = weightcalcdata.startindex
    size = weightcalcdata.testsize

    cause_dellist = []
    affected_dellist = []
    for index in range(vardims):
        if index not in weightcalcdata.causevarindexes:
            cause_dellist.append(index)
            logging.info("Deleted column " + str(index))
        if index not in weightcalcdata.affectedvarindexes:
            affected_dellist.append(index)
            logging.info("Deleted row " + str(index))

    if weightcalcdata.connections_used:
        newconnectionmatrix = weightcalcdata.connectionmatrix
    else:
        newconnectionmatrix = np.ones((vardims, vardims))
    # Substitute columns not used with zeros in connectionmatrix
    for cause_delindex in cause_dellist:
        newconnectionmatrix[:, cause_delindex] = np.zeros(vardims)
    # Substitute rows not used with zeros in connectionmatrix
    for affected_delindex in affected_dellist:
        newconnectionmatrix[affected_delindex, :] = np.zeros(vardims)

    # Initiate headerline for weightstore file
    # Create "Delay" as header for first row
    headerline = ['Delay']
    for affectedvarindex in weightcalcdata.affectedvarindexes:
        affectedvarname = weightcalcdata.variables[affectedvarindex]
        headerline.append(affectedvarname)

    # Define filename structure for CSV file containing weights between
    # a specific causevar and all the subsequent affectedvars
    def filename(weightname, boxindex, causevar):
        boxstring = 'box{:03d}'.format(boxindex)

        filedir = config_setup.ensure_existence(
            os.path.join(weightstoredir, weightname, boxstring), make=True)

        filename = '{}.csv'.format(causevar)

        return os.path.join(filedir, filename)

    # Store the weight calculation results in similar format as original data

    # Define weightstoredir up to the method level
    weightstoredir = config_setup.ensure_existence(
        os.path.join(weightcalcdata.saveloc, 'weightdata',
                     weightcalcdata.casename,
                     scenario, method, sigstatus, embedstatus), make=True)

    if weightcalcdata.single_entropies:
        # Initiate headerline for single signal entropies storage file
        signalent_headerline = weightcalcdata.variables
        # Define filename structure for CSV file

        def signalent_filename(name, boxindex):
            return signalent_filename_template.format(
                weightcalcdata.casename, scenario, name, boxindex)

        signalentstoredir = config_setup.ensure_existence(
            os.path.join(weightcalcdata.saveloc, 'signal_entropies'),
            make=True)

        signalent_filename_template = \
            os.path.join(signalentstoredir, '{}_{}_{}_box{:03d}.csv')

    # Generate boxes to use
    boxes = data_processing.split_tsdata(
        weightcalcdata.inputdata,
        weightcalcdata.sampling_rate * weightcalcdata.sub_sampling_interval,
        weightcalcdata.boxsize,
        weightcalcdata.boxnum)

    for boxindex in weightcalcdata.boxindexes:
        box = boxes[boxindex]

        # Calculate single signal entropies - do not worry about
        # delays, but still do it according to different boxes
        if weightcalcdata.single_entropies:
            # Calculate single signal entropies of all variables
            # and save output in similar format to
            # standard weight calculation results
            signalentlist = []
            for varindex, _ in enumerate(weightcalcdata.variables):
                vardata = box[:, varindex][startindex:startindex+size]
                entropy = data_processing.calc_signalent(vardata,
                                                         weightcalcdata)
                signalentlist.append(entropy)

            # Write the signal entropies to file - one file for each box
            # Each file will only have one line as we are not
            # calculating for different delays as is done for the case of
            # variable pairs.

            # Need to add another axis to signalentlist in order to make
            # it a sequence so that it can work with writecsv_weightcalc
            signalentlist = np.asarray(signalentlist)
            signalentlist = \
                signalentlist[np.newaxis, :]

            writecsv_weightcalc(signalent_filename(
                'signal_entropy',
                boxindex+1),
                signalentlist, signalent_headerline)

        # Start parallelising code here
        # Create one process for each causevarindex

        ###########################################################

        non_iter_args = [
            weightcalcdata, weightcalculator,
            box, startindex, size,
            newconnectionmatrix,
            method, boxindex,
            filename, headerline]

        # Run the script that will handle multiprocessing
        gaincalc_oneset.run(non_iter_args,
                            weightcalcdata.do_multiprocessing)

        ########################################################

    return None


def weightcalc(mode, case, writeoutput=False, single_entropies=False,
               fftcalc=False, do_multiprocessing=False):
    """Reports the maximum weight as well as associated delay
    obtained by shifting the affected variable behind the causal variable a
    specified set of delays.

    Supports calculating weights according to either correlation or transfer
    entropy metrics.

    """

    weightcalcdata = WeightcalcData(mode, case, single_entropies, fftcalc,
                                    do_multiprocessing)

    for scenario in weightcalcdata.scenarios:
        logging.info("Running scenario {}".format(scenario))
        # Update scenario-specific fields of weightcalcdata object
        weightcalcdata.scenariodata(scenario)
        for settings_name in weightcalcdata.settings_set:
            weightcalcdata.setsettings(scenario, settings_name)
            logging.info("Now running settings {}".format(settings_name))

            for method in weightcalcdata.methods:
                logging.info("Method: " + method)

                start_time = time.clock()
                calc_weights(weightcalcdata, method, scenario)
                end_time= time.clock()
                print end_time - start_time

if __name__ == '__main__':
    multiprocessing.freezeSupport()
