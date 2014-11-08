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
import jpype

# Own libraries
import config_setup
import data_processing

# Gain calculators
from gaincalculators import (PartialCorrWeightcalc, CorrWeightcalc,
                             TransentWeightcalc)

import datagen


class WeightcalcData:
    """Creates a data object from file and or function definitions for use in
    weight calculation methods.

    """
    def __init__(self, mode, case, single_entropies, fftcalc):
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

        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(),
                           "-Xms32M",
                           "-Xmx512M",
                           "-ea",
                           "-Djava.class.path=" + infodynamicsloc)

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
        settings_name = self.caseconfig[scenario]['settings']
        self.connections_used = (self.caseconfig[settings_name]
                                 ['use_connections'])
        bandgap_filtering = self.caseconfig[scenario]['bandgap_filtering']
        self.transient = self.caseconfig[settings_name]['transient']

        self.normalize = self.caseconfig[settings_name]['normalize']
        self.sigtest = self.caseconfig[settings_name]['sigtest']
        self.allthresh = self.caseconfig[settings_name]['allthresh']

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

        elif self.datatype == 'function':
            raw_tsdata_gen = self.caseconfig[scenario]['datagen']
            connectionloc = self.caseconfig[scenario]['connections']
            # TODO: Store function arguments in scenario config file
            samples = self.caseconfig[settings_name]['gensamples']
            func_delay = self.caseconfig[settings_name]['delay']
            # Get inputdata
            self.inputdata_raw = eval('datagen.' + raw_tsdata_gen)(samples,
                                                                   func_delay)
            # Get the variables and connection matrix
            self.connectionmatrix = eval('datagen.' + connectionloc)()
            self.startindex = 0

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

        if self.normalize:
            # Normalise (mean centre and variance scale) the input data
            self.inputdata_normstep = \
                data_processing.normalise_data(raw_tsdata, self.inputdata_raw,
                                               self.saveloc, self.casename,
                                               scenario)
        else:
            # Still norm centre the data
            self.inputdata_normstep = data_processing.subtract_mean(
                self.inputdata_raw)

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
        sub_sampling_interval = \
            self.caseconfig[settings_name]['sub_sampling_interval']
        # TODO: Use proper pandas.tseries.resample techniques
        # if it will really add any functionality
        self.inputdata = self.inputdata_originalrate[0::sub_sampling_interval]

        if self.transient:
            self.boxnum = self.caseconfig[settings_name]['boxnum']
            self.boxsize = self.caseconfig[settings_name]['boxsize']
        else:
            self.boxnum = 1  # Only a single box will be used
            self.boxsize = self.inputdata.shape[0] * \
                self.sampling_rate
            # This box should now return the same size
            # as the original data file - but it does not play a role at all
            # in the actual for the case of boxnum = 1

        # Select which of the boxes to evaluate
        if self.transient:
            self.boxindexes = self.caseconfig[scenario]['boxindexes']
            if self.boxindexes == 'all':
                self.boxindexes = range(len(self.boxindexes))
        else:
            self.boxindexes = 'all'

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


def calc_weights(weightcalcdata, method, scenario):
    """Determines the maximum weight between two variables by searching through
    a specified set of delays.

    method can be either 'cross_correlation' or 'transfer_entropy'

    """
    # Switch to calculate significance values at each data point and store in
    # files similar to weight calculations

    if method == 'cross_correlation':
        weightcalculator = CorrWeightcalc(weightcalcdata)
    elif method == 'transfer_entropy_kernel':
        weightcalculator = TransentWeightcalc(weightcalcdata, 'kernel')
    elif method == 'transfer_entropy_kraskov':
        weightcalculator = TransentWeightcalc(weightcalcdata, 'kraskov')
    elif method == 'partial_correlation':
        weightcalculator = PartialCorrWeightcalc(weightcalcdata)

    if weightcalcdata.sigtest:
        sigstatus = 'sigtested'
    elif not weightcalcdata.sigtest:
        sigstatus = 'nosigtest'

    vardims = len(weightcalcdata.variables)
    startindex = weightcalcdata.startindex
    size = weightcalcdata.testsize
    data_header = weightcalculator.data_header

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
        newconnectionmatrix[affected_dellist, :] = np.zeros(vardims)

    # Initiate headerline for weightstore file
    headerline = []
    # Create "Delay" as header for first row
    headerline.append('Delay')

    for affectedvarindex in weightcalcdata.affectedvarindexes:
            affectedvarname = weightcalcdata.variables[affectedvarindex]
            headerline.append(affectedvarname)

    # Define filename structure for CSV file containing weights between
    # a specific causevar and all the subsequent affectedvars
    def filename(name, method, boxindex, sigstatus, causevar):
        return filename_template.format(weightcalcdata.casename,
                                        scenario, name, method, sigstatus,
                                        boxindex, causevar)

    def sig_filename(name, method, boxindex, causevar):
        return sig_filename_template.format(weightcalcdata.casename,
                                            scenario, name, method,
                                            boxindex, causevar)

    # Store the weight calculation results in similar format as original data
    def writecsv_weightcalc(filename, items, header):
        """CSV writer customized for use in weightcalc function."""
        with open(filename, 'wb') as f:
            csv.writer(f).writerow(header)
            csv.writer(f).writerows(items)

    weightstoredir = config_setup.ensure_existance(
        os.path.join(weightcalcdata.saveloc, 'weightdata'), make=True)

    filename_template = os.path.join(weightstoredir,
                                     '{}_{}_{}_{}_{}_box{:03d}_{}.csv')

    sig_filename_template = os.path.join(weightstoredir,
                                         '{}_{}_{}_{}_box{:03d}_{}.csv')

    if weightcalcdata.single_entropies:
        # Initiate headerline for single signal entropies storage file
        signalent_headerline = weightcalcdata.variables
        # Define filename structure for CSV file

        def signalent_filename(name, boxindex):
            return signalent_filename_template.format(
                weightcalcdata.casename, scenario, name, boxindex)

        signalentstoredir = config_setup.ensure_existance(
            os.path.join(weightcalcdata.saveloc, 'signal_entopries'),
            make=True)

        signalent_filename_template = \
            os.path.join(signalentstoredir, '{}_{}_{}_box{:03d}.csv')

    # Generate boxes to use
    boxes = data_processing.split_tsdata(weightcalcdata.inputdata,
                                         weightcalcdata.sampling_rate,
                                         weightcalcdata.boxsize,
                                         weightcalcdata.boxnum)

    # Create storage lists weight_arrays, delay_arrays and datastores
    # that will be generated for each box

    weight_arrays = []
    delay_arrays = []
    datastores = []

    for boxindex in weightcalcdata.boxindexes:
        box = boxes[boxindex]

#    for boxindex, box in enumerate(boxes):

        weight_array = np.empty((vardims, vardims))
        delay_array = np.empty((vardims, vardims))
        weight_array[:] = np.NAN
        delay_array[:] = np.NAN
        datastore = []

        # Calculate single signal entropies - do not worry about
        # delays, but still do it according to different boxes
        if weightcalcdata.single_entropies:
                # Calculate single signal entropies of all variables
                # and save output in similar format to
                # standard weight calculation results
            signalentlist = []
            for varindex, variable in enumerate(weightcalcdata.variables):
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

        for causevarindex in weightcalcdata.causevarindexes:
            causevar = weightcalcdata.variables[causevarindex]

            # Initiate datalines with delays
            datalines_directional = \
                np.asarray(weightcalcdata.actual_delays)
            datalines_directional = datalines_directional[:, np.newaxis]
            datalines_absolute = datalines_directional.copy()
            datalines_neutral = datalines_directional.copy()
            # Datalines needed to store significance threshold values
            # for each variable combination
            datalines_sigthresh_directional = datalines_directional.copy()
            datalines_sigthresh_absolute = datalines_directional.copy()
            datalines_sigthresh_neutral = datalines_directional.copy()

            for affectedvarindex in weightcalcdata.affectedvarindexes:
                affectedvar = weightcalcdata.variables[affectedvarindex]
                logging.info("Analysing effect of: " + causevar + " on " +
                             affectedvar + " for box number: " +
                             str(boxindex + 1))

                if not(newconnectionmatrix[affectedvarindex,
                                           causevarindex] == 0):
                    weightlist = []
                    directional_weightlist = []
                    absolute_weightlist = []
                    sigthreshlist = []
                    directional_sigthreshlist = []
                    absolute_sigthreshlist = []

                    for delay in weightcalcdata.sample_delays:
                        logging.info("Now testing delay: " + str(delay))

                        causevardata = \
                            (box[:, causevarindex]
                                [startindex:startindex+size])

                        affectedvardata = \
                            (box[:, affectedvarindex]
                                [startindex+delay:startindex+size+delay])

                        weight = weightcalculator.calcweight(causevardata,
                                                             affectedvardata,
                                                             weightcalcdata,
                                                             causevarindex,
                                                             affectedvarindex)

                        # Calculate significance thresholds at each data point
                        if weightcalcdata.allthresh:
                            sigthreshold = \
                                weightcalculator.calcsigthresh(affectedvardata,
                                                               causevardata)

                        if len(weight) > 1:
                            # If weight contains directional as well as
                            # absolute weights, write to separate lists
                            directional_weightlist.append(weight[0])
                            absolute_weightlist.append(weight[1])
                            # Same approach with significance thresholds
                            if weightcalcdata.allthresh:
                                directional_sigthreshlist.append(
                                    sigthreshold[0])
                                absolute_sigthreshlist.append(
                                    sigthreshold[1])

                        else:
                            weightlist.append(weight[0])
                            if weightcalcdata.allthresh:
                                sigthreshlist.append(sigthreshold[0])

                    directional_name = 'weights_directional'
                    absolute_name = 'weights_absolute'
                    neutral_name = 'weights'
                    # Provide names for the significance threshold file types
                    if weightcalcdata.allthresh:
                        sig_directional_name = 'sigthresh_directional'
                        sig_absolute_name = 'sigthresh_absolute'
                        sig_neutral_name = 'sigthresh'

                    if len(weight) > 1:
                        weightlist = [directional_weightlist,
                                      absolute_weightlist]

                        # Combine weight data
                        weights_thisvar_directional = np.asarray(weightlist[0])
                        weights_thisvar_directional = \
                            weights_thisvar_directional[:, np.newaxis]

                        weights_thisvar_absolute = np.asarray(weightlist[1])
                        weights_thisvar_absolute = \
                            weights_thisvar_absolute[:, np.newaxis]

                        datalines_directional = \
                            np.concatenate((datalines_directional,
                                            weights_thisvar_directional),
                                           axis=1)

                        datalines_absolute = \
                            np.concatenate((datalines_absolute,
                                            weights_thisvar_absolute),
                                           axis=1)

                        writecsv_weightcalc(filename(
                            directional_name,
                            method, boxindex+1, sigstatus, causevar),
                            datalines_directional, headerline)

                        writecsv_weightcalc(filename(
                            absolute_name,
                            method, boxindex+1, sigstatus, causevar),
                            datalines_absolute, headerline)

                        # Do the same for the significance threshold
                        if weightcalcdata.allthresh:
                            sigthreshlist = [directional_sigthreshlist,
                                             absolute_sigthreshlist]

                            sigthresh_thisvar_directional = \
                                np.asarray(sigthreshlist[0])
                            sigthresh_thisvar_directional = \
                                sigthresh_thisvar_directional[:, np.newaxis]

                            sigthresh_thisvar_absolute = \
                                np.asarray(sigthreshlist[1])
                            sigthresh_thisvar_absolute = \
                                sigthresh_thisvar_absolute[:, np.newaxis]

                            datalines_sigthresh_directional = np.concatenate(
                                (datalines_sigthresh_directional,
                                 sigthresh_thisvar_directional),
                                axis=1)

                            datalines_sigthresh_absolute = \
                                np.concatenate((datalines_sigthresh_absolute,
                                                sigthresh_thisvar_absolute),
                                               axis=1)

                            writecsv_weightcalc(sig_filename(
                                sig_directional_name,
                                method, boxindex+1, causevar),
                                datalines_sigthresh_directional, headerline)

                            writecsv_weightcalc(sig_filename(
                                sig_absolute_name,
                                method, boxindex+1, causevar),
                                datalines_sigthresh_absolute, headerline)

                    else:
                        weights_thisvar_neutral = np.asarray(weightlist)
                        weights_thisvar_neutral = \
                            weights_thisvar_neutral[:, np.newaxis]

                        datalines_neutral = \
                            np.concatenate((datalines_neutral,
                                            weights_thisvar_neutral), axis=1)

                        writecsv_weightcalc(filename(
                            neutral_name,
                            method, boxindex+1, sigstatus, causevar),
                            datalines_neutral, headerline)

                        # Write the significance thresholds to file
                        if weightcalcdata.allthresh:
                            sigthresh_thisvar_neutral = np.asarray(weightlist)
                            sigthresh_thisvar_neutral = \
                                sigthresh_thisvar_neutral[:, np.newaxis]

                            datalines_sigthresh_neutral = \
                                np.concatenate((datalines_sigthresh_neutral,
                                                sigthresh_thisvar_neutral),
                                               axis=1)

                            writecsv_weightcalc(sig_filename(
                                sig_neutral_name,
                                method, boxindex+1, causevar),
                                datalines_sigthresh_neutral, headerline)

                    # Generate and store report files according to each method
                    [weight_array, delay_array, datastore] = \
                        weightcalculator.report(weightcalcdata, causevarindex,
                                                affectedvarindex, weightlist,
                                                weight_array, delay_array,
                                                datastore)

        # Delete entries from weightcalc matrix not used
        # Delete all rows and columns listed in affected_dellist, cause_dellist
        # from weight_array
        # Axis 0 is rows, axis 1 is columns
#        weight_array = np.delete(weight_array, cause_dellist, 1)
#        weight_array = np.delete(weight_array, affected_dellist, 0)

        # Do the same for delay_array
#        delay_array = np.delete(delay_array, cause_dellist, 1)
#        delay_array = np.delete(delay_array, affected_dellist, 0)

        weight_arrays.append(weight_array)
        delay_arrays.append(delay_array)
        datastores.append(datastore)

    return weight_arrays, delay_arrays, datastores, data_header


def writecsv_weightcalc(filename, items, header):
    """CSV writer customized for use in weightcalc function."""

    with open(filename, 'wb') as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(items)


def weightcalc(mode, case, writeoutput=False, single_entropies=False,
               fftcalc=False):
    """Reports the maximum weight as well as associated delay
    obtained by shifting the affected variable behind the causal variable a
    specified set of delays.

    Supports calculating weights according to either correlation or transfer
    entropy metrics.

    """

    weightcalcdata = WeightcalcData(mode, case, single_entropies, fftcalc)

    # Define export directories and filenames
    weightdir = config_setup.ensure_existance(os.path.join(
        weightcalcdata.saveloc, 'weightcalc'), make=True)

    filename_template = os.path.join(weightdir, '{}_{}_{}_{}.csv')

    def filename(method, name):
        return filename_template.format(case, scenario,
                                        method, name)

    maxweight_array_name = 'maxweight_array_box{:03d}'
    delay_array_name = 'delay_array_box{:03d}'
    weightcalc_data_name = 'weightcalc_data_box{:03d}'

    for scenario in weightcalcdata.scenarios:
        logging.info("Running scenario {}".format(scenario))
        # Update scenario-specific fields of weightcalcdata object
        weightcalcdata.scenariodata(scenario)

        for method in weightcalcdata.methods:
            logging.info("Method: " + method)

            # Test whether the 'weightcalc_data_box01' file already exists
            testlocation = filename(method, 'weightcalc_data_box001')
            if not os.path.exists(testlocation):
                # Continue with execution
                [weight_arrays, delay_arrays, datastores, data_header] = \
                    calc_weights(weightcalcdata, method, scenario)

                # TODO: Call this code on each weight array as soon as its
                # calculation is finished.

                for boxindex in weightcalcdata.boxindexes:
                    if writeoutput:
                        # Write arrays to file
                        np.savetxt(
                            filename(method,
                                     maxweight_array_name.format(boxindex+1)),
                            weight_arrays[boxindex],
                            delimiter=',')
                        np.savetxt(
                            filename(method,
                                     delay_array_name.format(boxindex+1)),
                            delay_arrays[boxindex],
                            delimiter=',')
                        # Write datastore to file
                        writecsv_weightcalc(
                            filename(method,
                                     weightcalc_data_name.format(boxindex+1)),
                            datastores[boxindex],
                            data_header)
            else:
                logging.info("The requested results are in existence")
