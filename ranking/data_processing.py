# -*- coding: utf-8 -*-
"""Data processing support tasks.

"""

import csv
import json
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import sklearn.preprocessing
import tables as tb
from numba import jit
from scipy import signal

import config_setup
from ranking import gaincalc
import transentropy


@jit
def shuffle_data(input_data):
    """Returns a (seeded) randomly shuffled array of data.
    The data input needs to be a two-dimensional numpy array.

    """

    shuffled = np.random.permutation(input_data)

    shuffled_formatted = np.zeros((1, len(shuffled)))
    shuffled_formatted[0, :] = shuffled

    return shuffled_formatted


def getfolders(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)

            break

    folders.reverse()

    return folders


@jit
def gen_iaaft_surrogates(data, iterations):
    """Generates iAAFT surrogates

    """
    # Make copy to  prevent rotation of array
    data_f = data.copy()
#    start_time = time.clock()
    xs = data_f.copy()
    # sorted amplitude stored
    xs.sort()
    # amplitude of fourier transform of orig
    pwx = np.abs(np.fft.fft(data_f))

    data_f.shape = (-1, 1)
    # random permutation as starting point
    xsur = np.random.permutation(data_f)
    xsur.shape = (1, -1)

    for i in range(iterations):
        fftsurx = pwx*np.exp(1j*np.angle(np.fft.fft(xsur)))
        xoutb = np.real(np.fft.ifft(fftsurx))
        ranks = xoutb.argsort(axis=1)
        xsur[:, ranks] = xs

#    end_time = time.clock()
#    logging.info("Time to generate surrogates: " + str(end_time - start_time))

    return(xsur)


class ResultReconstructionData:
    """Creates a data object from file and or function definitions for use in
    array creation methods.

    """

    def __init__(self, mode, case):

        # Get locations from configuration file
        self.saveloc, self.caseconfigloc, self.casedir, _ = \
            config_setup.runsetup(mode, case)
        # Load case config file
        self.caseconfig = json.load(
            open(os.path.join(self.caseconfigloc, case +
                              '_resultreconstruction' + '.json')))

        # Get data type
        self.datatype = self.caseconfig['datatype']

        self.case = case

        # Get scenarios
        self.scenarios = self.caseconfig['scenarios']
        self.case = case

    def scenariodata(self, scenario):
        """Retrieves data particular to each scenario for the case being
        investigated.

        """

        scenario_config = self.caseconfig[scenario]

        if scenario_config:
            if self.datatype == 'file':
                if 'bias_correction' in scenario_config:
                    self.bias_correction = scenario_config['bias_correction']
        else:
            settings = {}
            self.bias_correction = False
            logging.info("Defaulting to no bias correction")


def process_auxfile(filename, bias_correct=True, allow_neg=False):
    """Processes an auxfile and returns a list of affected_vars,
    weight_array as well as relative significance weight array.

    Parameters:
        filename (string): path to auxfile to process
        allow_neg (bool): if true, allows negative values in final weight arrays, otherwise sets them to zero.
        bias_correct (bool): if true, subtracts the mean of the null distribution off the final value in weight array

    """

    affectedvars = []
    weights = []
    nosigtest_weights = []
    sigweights = []
    delays = []
    sigthresholds = []
    #sigstds = []

    with open(filename, 'r') as auxfile:
        auxfilereader = csv.reader(auxfile, delimiter=',')
        for rowindex, row in enumerate(auxfilereader):
            if rowindex == 0:
                # Find the indices of important rows
                affectedvar_index = row.index('affectedvar')

                if 'max_ent' in row:
                    maxval_index = row.index('max_ent')
                else:
                    maxval_index = row.index('max_corr')

                if 'bias_mean' in row:
                    biasmean_index = row.index('bias_mean')
                else:
                    biasmean_index = None

                if 'threshold' in row:
                    thresh_index = row.index('threshold')
                else:
                    thresh_index = row.index('threshcorr')

                threshpass_index = row.index('threshpass')
                directionpass_index = row.index('directionpass')
                maxdelay_index = row.index('max_delay')

            if rowindex > 0:

                affectedvars.append(row[affectedvar_index])

                # Test if weight failed threshpass or directionpass test and
                # write as zero if true

                # In rare cases it might be desired to allow negative values
                # (e.g. correlation tests)
                # TODO: Put the allow_neg parameter in a configuration file
                # NOTE: allow_neg also removes significance testing

                weight_candidate = float(row[maxval_index])

                if allow_neg:
                    nosigtest_weight = weight_candidate
                    sigtest_weight = weight_candidate
                else:
                    if weight_candidate > 0.:
                        # Attach to no significance test result
                        nosigtest_weight = weight_candidate
                        if (row[threshpass_index] == 'False' or
                                row[directionpass_index] == 'False'):
                            sigtest_weight = 0.
                        else:
                            # threshpass is either None or True
                            sigtest_weight = weight_candidate
                    else:
                        sigtest_weight = 0.
                        nosigtest_weight = 0.

                # Perform bias correction if required
                if bias_correct and biasmean_index and (sigtest_weight > 0.):
                    sigtest_weight = sigtest_weight - float(row[biasmean_index])
                    if sigtest_weight < 0:
                        raise ValueError('Negative weight after subtracting biasmean')

                weights.append(sigtest_weight)
                nosigtest_weights.append(nosigtest_weight)
                delays.append(float(row[maxdelay_index]))

                try:
                    threshold = float(row[thresh_index])
                except:
                    threshold = 0.
                sigthresholds.append(threshold)

                # Test if sigtest passed before assigning weight
                if (row[threshpass_index] == 'True' and
                        row[directionpass_index] == 'True'):
                    # If the threshold is negative, take the absolute value
                    # TODO: Need to think the implications of this through
                    if threshold != 0:
                        sigweight = (float(row[maxval_index]) /
                                     abs(threshold))
                        if sigweight > 0.:
                            sigweights.append(sigweight)
                        else:
                            sigweights.append(0.)
                    else:
                        sigweights.append(0.)

                else:
                    sigweights.append(0.)

    return affectedvars, weights, nosigtest_weights, sigweights, delays, sigthresholds


def create_arrays(datadir, variables, bias_correct, generate_diffs):
    """
    datadir is the location of the auxdata and weights folders for the
    specific case that is under investigation

    variables is the list of variables

    """

    absoluteweightarray_name = 'weight_absolute_arrays'
    directionalweightarray_name = 'weight_directional_arrays'
    neutralweightarray_name = 'weight_arrays'
    difabsoluteweightarray_name = 'dif_weight_absolute_arrays'
    difdirectionalweightarray_name = 'dif_weight_directional_arrays'
    difneutralweightarray_name = 'dif_weight_arrays'
    absolutesigweightarray_name = 'sigweight_absolute_arrays'
    directionalsigweightarray_name = 'sigweight_directional_arrays'
    neutralsigweightarray_name = 'sigweight_arrays'
    absolutedelayarray_name = 'delay_absolute_arrays'
    directionaldelayarray_name = 'delay_directional_arrays'
    neutraldelayarray_name = 'delay_arrays'
    absolutesigthresholdarray_name = 'sigthreshold_absolute_arrays'
    directionalsigthresholdarray_name = 'sigthreshold_directional_arrays'
    neutralsigthresholdarray_name = 'sigthreshold_arrays'

    directories = next(os.walk(datadir))[1]

    test_strings = ['auxdata_absolute', 'auxdata_directional', 'auxdata']

    for test_string in test_strings:

        if test_string in directories:

            if test_string == 'auxdata_absolute':
                weightarray_name = absoluteweightarray_name
                difweightarray_name = difabsoluteweightarray_name
                sigweightarray_name = absolutesigweightarray_name
                delayarray_name = absolutedelayarray_name
                sigthresholdarray_name = absolutesigthresholdarray_name
            elif test_string == 'auxdata_directional':
                weightarray_name = directionalweightarray_name
                difweightarray_name = difdirectionalweightarray_name
                sigweightarray_name = directionalsigweightarray_name
                delayarray_name = directionaldelayarray_name
                sigthresholdarray_name = directionalsigthresholdarray_name
            elif test_string == 'auxdata':
                weightarray_name = neutralweightarray_name
                difweightarray_name = difneutralweightarray_name
                sigweightarray_name = neutralsigweightarray_name
                delayarray_name = neutraldelayarray_name
                sigthresholdarray_name = neutralsigthresholdarray_name

            boxes = next(os.walk(os.path.join(datadir, test_string)))[1]
            for box in boxes:
                boxdir = os.path.join(datadir, test_string, box)
                # Get list of causevars
                causevar_filenames = next(os.walk(boxdir))[2]
                causevars = []
                affectedvar_array = []
                weight_array = []
                nosigtest_weight_array = []
                sigweight_array = []
                delay_array = []
                sigthreshold_array = []
                for causevar_file in causevar_filenames:
                    causevars.append(str(causevar_file[:-4]))

                    # Open auxfile and return weight array as well as
                    # significance relative weight arrays

                    # TODO: Confirm whether correlation tests absolutes correlations before sending to auxfile
                    # Otherwise, the allow null must be used much more wisely
                    (affectedvars, weights, nosigtest_weights,
                     sigweights, delays, sigthresholds) = \
                        process_auxfile(os.path.join(boxdir, causevar_file), bias_correct=bias_correct)

                    affectedvar_array.append(affectedvars)
                    weight_array.append(weights)
                    nosigtest_weight_array.append(nosigtest_weights)
                    sigweight_array.append(sigweights)
                    delay_array.append(delays)
                    sigthreshold_array.append(sigthresholds)

                # Write the arrays to file
                # Create a base array based on the full set of variables
                # found in the typical weightcalcdata function

                # Initialize matrix with variables written
                # in first row and column
                weights_matrix = np.zeros(
                    (len(variables)+1, len(variables)+1)).astype(object)

                weights_matrix[0, 0] = ''
                weights_matrix[0, 1:] = variables
                weights_matrix[1:, 0] = variables

                nosigtest_weights_matrix = np.copy(weights_matrix)
                sigweights_matrix = np.copy(weights_matrix)
                delay_matrix = np.copy(weights_matrix)
                sigthresh_matrix = np.copy(weights_matrix)

                # Write results to appropriate entries in array
                for causevar_index, causevar in enumerate(causevars):
                    causevarloc = variables.index(causevar)
                    for affectedvar_index, affectedvar in \
                            enumerate(affectedvar_array[causevar_index]):
                        affectedvarloc = variables.index(affectedvar)

                        weights_matrix[affectedvarloc+1, causevarloc+1] = \
                            weight_array[causevar_index][affectedvar_index]
                        nosigtest_weights_matrix[affectedvarloc+1,
                                                 causevarloc+1] = \
                            nosigtest_weight_array[
                                causevar_index][affectedvar_index]
                        sigweights_matrix[affectedvarloc+1, causevarloc+1] = \
                            sigweight_array[causevar_index][affectedvar_index]
                        delay_matrix[affectedvarloc+1, causevarloc+1] = \
                            delay_array[causevar_index][affectedvar_index]
                        sigthresh_matrix[affectedvarloc+1, causevarloc+1] = \
                            sigthreshold_array[causevar_index][affectedvar_index]

                # Write to CSV files
                weightarray_dir = os.path.join(
                    datadir, weightarray_name, box)
                config_setup.ensure_existence(weightarray_dir)
                weightfilename = \
                    os.path.join(weightarray_dir, 'weight_array.csv')
                np.savetxt(weightfilename, weights_matrix,
                           delimiter=',', fmt='%s')

                delayarray_dir = os.path.join(
                    datadir, delayarray_name, box)
                config_setup.ensure_existence(delayarray_dir)
                delayfilename = \
                    os.path.join(delayarray_dir, 'delay_array.csv')
                np.savetxt(delayfilename, delay_matrix,
                           delimiter=',', fmt='%s')

                dirparts = getfolders(datadir)
                if 'sigtested' in dirparts:

                    dirparts[
                        dirparts.index('sigtested')] = 'nosigtest'
                    nosigtest_savedir = dirparts[0]
                    for pathpart in dirparts[1:]:
                        nosigtest_savedir = os.path.join(
                            nosigtest_savedir, pathpart)

                    nosigtest_weightarray_dir = os.path.join(
                        nosigtest_savedir, weightarray_name, box)
                    config_setup.ensure_existence(nosigtest_weightarray_dir)

                    nosigtest_delayarray_dir = os.path.join(
                        nosigtest_savedir, delayarray_name, box)
                    config_setup.ensure_existence(nosigtest_delayarray_dir)

                    delayfilename = \
                        os.path.join(nosigtest_delayarray_dir,
                                     'delay_array.csv')
                    np.savetxt(delayfilename, delay_matrix,
                               delimiter=',', fmt='%s')

                    weightfilename = \
                        os.path.join(nosigtest_weightarray_dir,
                                     'weight_array.csv')
                    np.savetxt(weightfilename, nosigtest_weights_matrix,
                               delimiter=',', fmt='%s')

                    sigweightarray_dir = os.path.join(
                        datadir, sigweightarray_name, box)
                    config_setup.ensure_existence(sigweightarray_dir)
                    sigweightfilename = \
                        os.path.join(sigweightarray_dir, 'sigweight_array.csv')
                    np.savetxt(sigweightfilename, sigweights_matrix,
                               delimiter=',', fmt='%s')

                    sigthresholdarray_dir = os.path.join(
                        datadir, sigthresholdarray_name, box)
                    config_setup.ensure_existence(sigthresholdarray_dir)
                    sigthresholdfilename = \
                        os.path.join(sigthresholdarray_dir, 'sigthreshold_array.csv')
                    np.savetxt(sigthresholdfilename, sigthresh_matrix,
                               delimiter=',', fmt='%s')

            if generate_diffs:
                boxes = list(boxes)
                boxes.sort()

                for boxindex, box in enumerate(boxes):

                    difweights_matrix = np.zeros(
                        (len(variables) + 1, len(variables) + 1)).astype(object)

                    difweights_matrix[0, 0] = ''
                    difweights_matrix[0, 1:] = variables
                    difweights_matrix[1:, 0] = variables

                    if boxindex > 0:

                        base_weight_array_dir = os.path.join(
                            datadir, weightarray_name, boxes[boxindex - 1])  # Already one behind
                        base_weight_array_filename = \
                            os.path.join(base_weight_array_dir, 'weight_array.csv')
                        final_weight_array_dir = os.path.join(
                            datadir, weightarray_name, box)
                        final_weight_array_filename = \
                            os.path.join(final_weight_array_dir, 'weight_array.csv')

                        with open(base_weight_array_filename, 'r') as f:
                            num_cols = len(f.readline().split(','))
                            f.seek(0)
                            base_weight_matrix = np.genfromtxt(f, usecols=range(1, num_cols), skip_header=1, delimiter=',')
                            f.close()

                        with open(final_weight_array_filename, 'r') as f:
                            num_cols = len(f.readline().split(','))
                            f.seek(0)
                            final_weight_matrix = np.genfromtxt(f, usecols=range(1, num_cols), skip_header=1, delimiter=',')
                            f.close()


                        # Calculate difference and save to file
                        # TODO: Investigate effect of taking absolute of differences
                        difweights_matrix[1:, 1:] = abs(final_weight_matrix) - abs(base_weight_matrix)

                    difweightarray_dir = os.path.join(
                        datadir, difweightarray_name, box)
                    config_setup.ensure_existence(difweightarray_dir)
                    difweightfilename = \
                        os.path.join(difweightarray_dir, 'dif_weight_array.csv')
                    np.savetxt(difweightfilename, difweights_matrix,
                               delimiter=',', fmt='%s')


                    if 'sigtested' in getfolders(datadir):

                        nosigtest_difweights_matrix = np.zeros(
                            (len(variables) + 1, len(variables) + 1)).astype(object)

                        nosigtest_difweights_matrix[0, 0] = ''
                        nosigtest_difweights_matrix[0, 1:] = variables
                        nosigtest_difweights_matrix[1:, 0] = variables

                        if boxindex > 0:

                            nosigtest_base_weight_array_dir = os.path.join(
                                nosigtest_savedir, weightarray_name, boxes[boxindex - 1])
                            nosigtest_base_weight_array_filename = \
                                os.path.join(nosigtest_base_weight_array_dir, 'weight_array.csv')
                            nosigtest_final_weight_array_dir = os.path.join(
                                nosigtest_savedir, weightarray_name, box)
                            nosigtest_final_weight_array_filename = \
                                os.path.join(nosigtest_final_weight_array_dir, 'weight_array.csv')

                            with open(nosigtest_base_weight_array_filename, 'r') as f:
                                num_cols = len(f.readline().split(','))
                                f.seek(0)
                                nosigtest_base_weight_matrix = np.genfromtxt(f, usecols=range(1, num_cols),
                                                                             skip_header=1, delimiter=',')
                                f.close()

                            with open(nosigtest_final_weight_array_filename, 'r') as f:
                                num_cols = len(f.readline().split(','))
                                f.seek(0)
                                nosigtest_final_weight_matrix = np.genfromtxt(f, usecols=range(1, num_cols),
                                                                    skip_header=1, delimiter=',')
                                f.close()

                            # Calculate difference and save to file
                            # TODO: Investigate effect of taking absolute of differences
                            nosigtest_difweights_matrix[1:, 1:] = abs(nosigtest_final_weight_matrix) - \
                                                                  abs(nosigtest_base_weight_matrix)

                        nosigtest_difweightarray_dir = os.path.join(
                            nosigtest_savedir, difweightarray_name, box)
                        config_setup.ensure_existence(nosigtest_difweightarray_dir)
                        nosigtest_difweightfilename = \
                            os.path.join(nosigtest_difweightarray_dir, 'dif_weight_array.csv')
                        np.savetxt(nosigtest_difweightfilename, nosigtest_difweights_matrix,
                                   delimiter=',', fmt='%s')


    return None


def create_signtested_directionalarrays(datadir, writeoutput):
    """Checks whether the directional weight arrays have corresponding
    absolute positive entries, writes another version with zeros if
    absolutes are negative.

    datadir is the location of the auxdata and weights folders for the
    specific case that is under investigation

    tsfilename is the file name of the original time series data file
    used to generate each case and is only used for generating a list of
    variables

    """

    signtested_weightarrayname = 'signtested_weight_directional_arrays'
    signtested_sigweightarrayname = 'signtested_sigweight_directional_arrays'

    directories = next(os.walk(datadir))[1]

    test_strings = ['weight_directional_arrays',
                    'sigweight_directional_arrays']

    lookup_strings = ['weight_absolute_arrays',
                      'sigweight_absolute_arrays']

    boxfilenames = {'weight_absolute_arrays': 'weight_array',
                    'weight_directional_arrays': 'weight_array',
                    'sigweight_absolute_arrays': 'sigweight_array',
                    'sigweight_directional_arrays': 'sigweight_array'}

    for test_index, test_string in enumerate(test_strings):
        if test_string in directories:

            if test_string == 'weight_directional_arrays':
                signtested_directionalweightarrayname = \
                    signtested_weightarrayname
            if test_string == 'sigweight_directional_arrays':
                signtested_directionalweightarrayname = \
                    signtested_sigweightarrayname

            boxes = next(os.walk(os.path.join(datadir, test_string)))[1]
            for box in boxes:
                dirboxdir = os.path.join(datadir, test_string, box)
                absboxdir = os.path.join(datadir, lookup_strings[test_index],
                                         box)

                # Read the contents of the test_string array
                dir_arraydf = pd.read_csv(
                    os.path.join(dirboxdir,
                                 boxfilenames[test_string] + '.csv'))
                # Read the contents of the comparative lookup_string array
                abs_arraydf = pd.read_csv(os.path.join(
                    absboxdir,
                    boxfilenames[lookup_strings[test_index]] + '.csv'))

                # Causevars is the first line of the array being read
                # Affectedvars is the first column of the array being read

                causevars = \
                    [dir_arraydf.columns[1:][i]
                     for i in range(0, len(dir_arraydf.columns[1:]))]

                affectedvars = \
                    [dir_arraydf[dir_arraydf.columns[0]][i]
                     for i in range(
                     0, len(dir_arraydf[dir_arraydf.columns[0]]))]

                # Create directional signtested array
                signtested_dir_array = np.zeros(
                    (len(affectedvars)+1, len(causevars)+1)).astype(object)

                # Initialize array with causevar labels in first column
                # and affectedvar labels in first row
                signtested_dir_array[0, 0] = ''
                signtested_dir_array[0, 1:] = causevars
                signtested_dir_array[1:, 0] = affectedvars

                for causevarindex in range(len(causevars)):
                    for affectedvarindex in range(len(affectedvars)):
                        # Check the sign of the abs_arraydf for this entry
                        abs_value = (
                            abs_arraydf[abs_arraydf.columns[causevarindex+1]]
                            [affectedvarindex])

                        if abs_value > 0:
                            signtested_dir_array[affectedvarindex+1,
                                                 causevarindex+1] = \
                                (dir_arraydf[
                                    dir_arraydf.columns[causevarindex+1]]
                                 [affectedvarindex])
                        else:
                            signtested_dir_array[affectedvarindex+1,
                                                 causevarindex+1] = 0

                # Write to CSV file
                if writeoutput:
                    signtested_weightarray_dir = os.path.join(
                        datadir, signtested_directionalweightarrayname, box)
                    config_setup.ensure_existence(signtested_weightarray_dir)

                    signtested_weightfilename = \
                        os.path.join(signtested_weightarray_dir,
                                     boxfilenames[test_string] + '.csv')
                    np.savetxt(signtested_weightfilename, signtested_dir_array,
                               delimiter=',', fmt='%s')

    return None


def extract_trends(datadir, writeoutput):
    """
    datadir is the location of the weight_array and delay_array folders for the
    specific case that is under investigation

    tsfilename is the file name of the original time series data file
    used to generate each case and is only used for generating a list of
    variables

    """

    # Create array to trend name dictionary
    namesdict = {'weight_absolute_arrays': 'weight_absolute_trend',
                 'weight_directional_arrays': 'weight_directional_trend',
                 'signtested_weight_directional_arrays':
                     'signtested_weight_directional_trend',
                 'weight_arrays': 'weight_trend',
                 'sigweight_absolute_arrays': 'sigweight_absolute_trend',
                 'sigweight_directional_arrays': 'sigweight_directional_trend',
                 'signtested_sigweight_directional_arrays':
                     'signtested_sigweight_directional_trend',
                 'sigweight_arrays': 'sigweight_trend',
                 'delay_absolute_arrays': 'delay_absolute_trend',
                 'delay_directional_arrays': 'delay_directional_trend',
                 'delay_arrays': 'delay_trend',
                 'sigthreshold_absolute_arrays': 'sigthreshold_absolute_trend',
                 'sigthreshold_directional_arrays': 'sigthreshold_directional_trend',
                 'sigthreshold_arrays': 'sigthreshold_trend'}

    boxfilenames = {'weight_absolute_arrays': 'weight_array',
                    'weight_directional_arrays': 'weight_array',
                    'signtested_weight_directional_arrays': 'weight_array',
                    'weight_arrays': 'weight_array',
                    'sigweight_absolute_arrays': 'sigweight_array',
                    'sigweight_directional_arrays': 'sigweight_array',
                    'signtested_sigweight_directional_arrays':
                        'sigweight_array',
                    'sigweight_arrays': 'sigweight_array',
                    'delay_absolute_arrays': 'delay_array',
                    'delay_directional_arrays': 'delay_array',
                    'delay_arrays': 'delay_array',
                    'sigthreshold_absolute_arrays': 'sigthreshold_array',
                    'sigthreshold_directional_arrays': 'sigthreshold_array',
                    'sigthreshold_arrays': 'sigthreshold_array'}

    directories = next(os.walk(datadir))[1]

    test_strings = namesdict.keys()

    savedir = change_dirtype(datadir, 'weightdata', 'trends')

    for test_string in test_strings:

        if test_string in directories:

            trendname = namesdict[test_string]

            arraydataframes = []

            boxes = next(os.walk(os.path.join(datadir, test_string)))[1]
            for box in boxes:
                boxdir = os.path.join(datadir, test_string, box)

                # Read the contents of the test_string array
                arraydf = pd.read_csv(
                    os.path.join(boxdir, boxfilenames[test_string] + '.csv'))

                arraydataframes.append(arraydf)

            # Causevars is the first line of the array being read
            # Affectedvars is the first column of the array being read

            arraydf = arraydataframes[0]

            causevars = [arraydf.columns[1:][i]
                         for i in range(0, len(arraydf.columns[1:]))]
            affectedvars = \
                [arraydf[arraydf.columns[0]][i]
                 for i in range(0, len(arraydf[arraydf.columns[0]]))]

            for causevar in causevars:
                # Create array with trends for specific causevar
                trend_array = np.zeros((len(affectedvars), len(boxes)+1),
                                       dtype=object)
                # Initialize array with affectedvar labels in first row
                trend_array[:, 0] = affectedvars

                for affectedvarindex in range(len(affectedvars)):
                    trendvalues = []
                    for arraydf in arraydataframes:
                        trendvalues.append(arraydf[causevar][affectedvarindex])
                    trend_array[affectedvarindex:, 1:] = trendvalues

                # Write to CSV file
                if writeoutput:
                    trend_dir = os.path.join(
                        savedir, causevar)
                    config_setup.ensure_existence(trend_dir)

                    trendfilename = \
                        os.path.join(trend_dir, trendname + '.csv')
                    np.savetxt(trendfilename, trend_array.T,
                               delimiter=',', fmt='%s')

    return None


def result_reconstruction(mode, case, writeoutput):
    """Reconstructs the weight_array and delay_array for different weight types
    from data generated by run_weightcalc process.

    WIP:
    For transient cases, generates difference arrays between boxes.

    The results are written to the same folders where the files are found.


    """

    resultreconstructiondata = ResultReconstructionData(mode, case)

    weightcalcdata = \
        gaincalc.WeightcalcData(mode, case, False, False, False)

    saveloc, caseconfigdir, _, _ = config_setup.runsetup(mode, case)

    caseconfig = json.load(
        open(os.path.join(caseconfigdir, case + '_weightcalc' + '.json')))

    # Directory where subdirectories for scenarios will be stored
    scenariosdir = os.path.join(saveloc, 'weightdata', case)

    # Get list of all scenarios
    scenarios = next(os.walk(scenariosdir))[1]

    for scenario in scenarios:
        print(scenario)

        resultreconstructiondata.scenariodata(scenario)

        weightcalcdata.setsettings(scenario,
                                   caseconfig[scenario]['settings'][0])

        methodsdir = os.path.join(scenariosdir, scenario)
        methods = next(os.walk(methodsdir))[1]
        for method in methods:
            print(method)
            sigtypesdir = os.path.join(methodsdir, method)
            sigtypes = next(os.walk(sigtypesdir))[1]
            for sigtype in sigtypes:
                print(sigtype)
                embedtypesdir = os.path.join(sigtypesdir, sigtype)
                embedtypes = next(os.walk(embedtypesdir))[1]
                for embedtype in embedtypes:
                    print(embedtype)
                    datadir = os.path.join(embedtypesdir, embedtype)
                    create_arrays(datadir, weightcalcdata.variables,
                                  resultreconstructiondata.bias_correction,
                                  weightcalcdata.generate_diffs)
                    # Provide directional array version tested with absolute
                    # weight sign
                    #create_signtested_directionalarrays(datadir, writeoutput)

    return None


def trend_extraction(mode, case, writeoutput):
    """Extracts dynamic trend of weights and delays out of weight_array
    and delay_array results between multiple boxes generated by the
    run_createarrays process for transient cases.

    The results are written to the trends results directory.

    """

    saveloc, _, _, _ = config_setup.runsetup(mode, case)

    # Directory where subdirectories for scenarios will be stored
    scenariosdir = os.path.join(saveloc, 'weightdata', case)

    # Get list of all scenarios
    scenarios = next(os.walk(scenariosdir))[1]

    for scenario in scenarios:
        print(scenario)

        methodsdir = os.path.join(scenariosdir, scenario)
        methods = next(os.walk(methodsdir))[1]
        for method in methods:
            print(method)
            sigtypesdir = os.path.join(methodsdir, method)
            sigtypes = next(os.walk(sigtypesdir))[1]
            for sigtype in sigtypes:
                print(sigtype)
                embedtypesdir = os.path.join(sigtypesdir, sigtype)
                embedtypes = next(os.walk(embedtypesdir))[1]
                for embedtype in embedtypes:
                    print(embedtype)
                    datadir = os.path.join(embedtypesdir, embedtype)
                    extract_trends(datadir, writeoutput)

    return None


def csv_to_h5(saveloc, raw_tsdata, scenario, case, overwrite=True):

    # Name the dataset according to the scenario
    dataset = scenario

    datapath = config_setup.ensure_existence(os.path.join(
        saveloc, 'data', case), make=True)

    filename = os.path.join(datapath, scenario + '.h5')

    if overwrite or (not os.path.exists(filename)):

        hdf5writer = tb.open_file(filename, 'w')
        data = np.genfromtxt(raw_tsdata, delimiter=',')
        # Strip time column and labels first row
        data = data[1:, 1:]
        array = hdf5writer.create_array(hdf5writer.root, dataset, data)

        array.flush()
        hdf5writer.close()

    return datapath


def read_timestamps(raw_tsdata):
    timestamps = []
    with open(raw_tsdata) as f:
        datareader = csv.reader(f)
        for rowindex, row in enumerate(datareader):
            if rowindex > 0:
                timestamps.append(row[0])
    timestamps = np.asarray(timestamps)
    return timestamps


def read_variables(raw_tsdata):
    with open(raw_tsdata) as f:
        variables = next(csv.reader(f))[1:]
    return variables


def writecsv(filename, items, header=None):
    """Write CSV directly"""
    with open(filename, 'w', newline='') as f:
        if header is not None:
            csv.writer(f).writerow(header)
        csv.writer(f).writerows(items)


def change_dirtype(datadir, oldtype, newtype):
    dirparts = getfolders(datadir)
    dirparts[dirparts.index(oldtype)] = newtype
    datadir = dirparts[0]
    for pathpart in dirparts[1:]:
        datadir = os.path.join(datadir, pathpart)

    return datadir


def fft_calculation(headerline, normalised_tsdata, variables, sampling_rate,
                    sampling_unit, saveloc, case, scenario,
                    plotting=False, plotting_endsample=500):

    # logging.info("Starting FFT calculations")
    # Using a print command instead as logging is late
    print("Starting FFT calculations")

    # Change first entry of headerline from "Time" to "Frequency"
    headerline[0] = 'Frequency'

    # Get frequency list (this is the same for all variables)
    freqlist = np.fft.rfftfreq(len(normalised_tsdata[:, 0]), sampling_rate)

    freqlist = freqlist[:, np.newaxis]

    fft_data = np.zeros((len(freqlist), len(variables)))

    def filename(name):
        return filename_template.format(case, scenario, name)

    for varindex in range(len(variables)):
        variable = variables[varindex]
        vardata = normalised_tsdata[:, varindex]

        # Compute FFT (normalised amplitude)
        var_fft = abs(np.fft.rfft(vardata)) * \
            (2. / len(vardata))

        fft_data[:, varindex] = var_fft

        if plotting:
            plt.figure()
            plt.plot(freqlist[0:plotting_endsample],
                     var_fft[0:plotting_endsample],
                     'r', label=variable)
            plt.xlabel('Frequency (1/' + sampling_unit + ')')
            plt.ylabel('Normalised amplitude')
            plt.legend()

            plotdir = config_setup.ensure_existence(
                os.path.join(saveloc, 'fftplots'), make=True)

            filename_template = os.path.join(plotdir,
                                             'FFT_{}_{}_{}.pdf')

            plt.savefig(filename(variable))
            plt.close()

#    varmaxindex = var_fft.tolist().index(max(var_fft))
#    print variable + " maximum signal strenght frequency: " + \
#        str(freqlist[varmaxindex])

    # Combine frequency list and FFT data
    datalines = np.concatenate((freqlist, fft_data), axis=1)

    # Define export directories and filenames
    datadir = config_setup.ensure_existence(
        os.path.join(saveloc, 'fftdata'), make=True)

    filename_template = os.path.join(datadir, '{}_{}_{}.csv')

    writecsv(filename('fft'), datalines, headerline)

    logging.info("Done with FFT calculations")
#    print "Done with FFT calculations"

    return None


def write_boxdates(boxdates, saveloc, case, scenario):

    def filename(name):
        return filename_template.format(case, scenario, name)

    datadir = config_setup.ensure_existence(
        os.path.join(saveloc, 'boxdates'), make=True)
    filename_template = os.path.join(datadir, '{}_{}_{}.csv')

    headerline = ['Box index', 'Box start', 'Box end']
    datalines = np.zeros((len(boxdates), 3))
    for index, boxdate in enumerate(boxdates):
        box_index = index + 1
        box_start = boxdate[0]
        box_end = boxdate[-1]
        datalines[index, :] = [box_index, box_start, box_end]

    writecsv(filename('boxdates'), datalines, headerline)

    return None


def bandgap(min_freq, max_freq, vardata):
    """Bandgap filter based on FFT/IFFT concatenation"""
    # TODO: Add buffer values in order to prevent ringing
    freqlist = np.fft.rfftfreq(vardata.size, 1)
    # Investigate effect of using abs()
    var_fft = np.fft.rfft(vardata)
    cut_var_fft = var_fft.copy()
    cut_var_fft[(freqlist < min_freq)] = 0
    cut_var_fft[(freqlist > max_freq)] = 0

    cut_vardata = np.fft.irfft(cut_var_fft)

    return cut_vardata


def bandgapfilter_data(raw_tsdata, normalised_tsdata, variables,
                       low_freq, high_freq,
                       saveloc, case, scenario):
    """Bandgap filter data between the specified high and low frequenices.
     Also writes filtered data to standard format for easy analysis in
     other software, for example TOPCAT.

     """

    # TODO: add two buffer indices to the start and end to eliminate ringing
    # Header and time from main source file
    headerline = np.genfromtxt(raw_tsdata, delimiter=',', dtype='string')[0, :]
    time = np.genfromtxt(raw_tsdata, delimiter=',')[1:, 0]
    time = time[:, np.newaxis]

    # Compensate for the fact that there is one less entry returned if the
    # number of samples is odd
    if bool(normalised_tsdata.shape[0] % 2):
        inputdata_bandgapfiltered = np.zeros((normalised_tsdata.shape[0]-1,
                                             normalised_tsdata.shape[1]))
    else:
        inputdata_bandgapfiltered = np.zeros_like(normalised_tsdata)

    for varindex in range(len(variables)):
        vardata = normalised_tsdata[:, varindex]
        bandgapped_vardata = bandgap(low_freq, high_freq, vardata)
        inputdata_bandgapfiltered[:, varindex] = bandgapped_vardata

    if bool(normalised_tsdata.shape[0] % 2):
        # Only write from the second time entry as there is one less datapoint
        # TODO: Currently it seems to exclude the last instead? Confirm
        datalines = np.concatenate((time[:-1],
                                    inputdata_bandgapfiltered), axis=1)
    else:
        datalines = np.concatenate((time, inputdata_bandgapfiltered), axis=1)

    # Define export directories and filenames
    datadir = config_setup.ensure_existence(
        os.path.join(saveloc, 'bandgappeddata'), make=True)

    filename_template = os.path.join(datadir, '{}_{}_{}_{}_{}.csv')

    def filename(name, lowfreq, highfreq):
        return filename_template.format(case, scenario, name,
                                        lowfreq, highfreq)

    # Store the normalised data in similar format as original data
    writecsv(filename('bandgapped_data', str(low_freq), str(high_freq)),
             datalines, headerline)

    return inputdata_bandgapfiltered


def detrend_linear_model(data):

    df = pd.DataFrame(data)
    detrended_df = pd.DataFrame(signal.detrend(df.dropna(), axis=0))
    detrended_df.index = df.dropna().index
    detrended_df.columns = df.columns

    return detrended_df.dropna().values


def detrend_first_differences(data):

    df = pd.DataFrame(data)
    detrended_df = df - df.shift(1)
    # Make first entry zero
    detrended_df.iloc[0, :] = 0.

    return detrended_df.dropna().values


def detrend_link_relatives(data):

    df = pd.DataFrame(data)
    detrended_df = df / df.shift(1)

    return detrended_df.dropna().values


def skogestad_scale_select(vartype, lower_limit, nominal_level, high_limit):
    if vartype == 'D':
        limit = max((nominal_level - lower_limit),
                    (high_limit - nominal_level))
    elif vartype == 'S':
        limit = min((nominal_level - lower_limit),
                    (high_limit - nominal_level))
    else:
        raise NameError("Variable type flag not recognized")
    return limit


def skogestad_scale(data_raw, variables, scalingvalues):
    if scalingvalues is None:
        raise ValueError("Scaling values not defined")

    data_skogestadscaled = np.zeros_like(data_raw)

    scalingvalues['scale_factor'] = map(
        skogestad_scale_select, scalingvalues['vartype'], scalingvalues['low'],
        scalingvalues['nominal'], scalingvalues['high'])

    # Loop through variables
    # The variables are aligned with the columns in raw_data
    for index, var in enumerate(variables):
        factor = scalingvalues.loc[var]['scale_factor']
        nominalval = scalingvalues.loc[var]['nominal']
        data_skogestadscaled[:, index] = \
            (data_raw[:, index] - nominalval) / factor

    return data_skogestadscaled


def write_normdata(saveloc, case, scenario, headerline, datalines):

    # Define export directories and filenames
    datadir = config_setup.ensure_existence(
        os.path.join(saveloc, 'normdata'), make=True)

    filename_template = os.path.join(datadir, '{}_{}_{}.csv')

    def filename(name):
        return filename_template.format(case, scenario, name)

    # Store the normalised data in similar format as original data
    writecsv(filename('normalised_data'), datalines, headerline)

    return None


def write_detrenddata(saveloc, case, scenario, headerline, datalines):

    # Define export directories and filenames
    datadir = config_setup.ensure_existence(
        os.path.join(saveloc, 'detrenddata'), make=True)

    filename_template = os.path.join(datadir, '{}_{}_{}.csv')

    def filename(name):
        return filename_template.format(case, scenario, name)

    # Store the detrended data in similar format as original data
    writecsv(filename('detrended_data'), datalines, headerline)

    return None


def normalise_data(headerline, timestamps, inputdata_raw, variables,
                   saveloc, case, scenario,
                   method, weight_methods, scalingvalues):

    if method == 'standardise':
        inputdata_normalised = \
            sklearn.preprocessing.scale(inputdata_raw, axis=0)
    elif method == 'skogestad':
        inputdata_normalised = \
            skogestad_scale(inputdata_raw, variables, scalingvalues)
    elif not method:
        # If method is simply false
        # Still mean center the data
        # This breaks when trying to use discrete methods
        if 'transfer_entropy_discrete' not in weight_methods:
            inputdata_normalised = subtract_mean(inputdata_raw)
        else:
            inputdata_normalised = inputdata_raw
    else:
        raise NameError("Normalisation method not recognized")

    datalines = np.concatenate((timestamps[:, np.newaxis],
                                inputdata_normalised), axis=1)

    write_normdata(saveloc, case, scenario, headerline, datalines)

    return inputdata_normalised


def detrend_data(headerline, timestamps, inputdata,
                 saveloc, case, scenario,
                 method):

    if method == 'first_differences':
        inputdata_detrended = \
            detrend_first_differences(inputdata)
    elif method == 'link_relatives':
        inputdata_detrended = \
            detrend_link_relatives(inputdata)
    elif method == 'linear_model':
        inputdata_detrended = \
            detrend_linear_model(inputdata)
    elif not method:
        # If method is False
        # Write received data without any modifications
        inputdata_detrended = inputdata

    else:
        raise NameError("Normalisation method not recognized")

    datalines = np.concatenate((timestamps[:, np.newaxis],
                                inputdata_detrended), axis=1)

    write_detrenddata(saveloc, case, scenario, headerline, datalines)

    return inputdata_detrended


def subtract_mean(inputdata_raw):
    """Subtracts mean from input data."""

    inputdata_lessmean = np.zeros_like(inputdata_raw)

    for col in range(inputdata_raw.shape[1]):
        colmean = np.mean(inputdata_raw[:, col])
        inputdata_lessmean[:, col] = \
            (inputdata_raw[:, col] - colmean)
    return inputdata_lessmean


def read_connectionmatrix(connection_loc):
    """Imports the connection scheme for the data.
    The format of the CSV file should be:
    empty space, var1, var2, etc... (first row)
    var1, value, value, value, etc... (second row)
    var2, value, value, value, etc... (third row)
    etc...

    value = 1 if column variable points to row variable (causal relationship)
    value = 0 otherwise

    """
    with open(connection_loc) as f:
        variables = csv.reader(f).next()[1:]
        connectionmatrix = np.genfromtxt(f, delimiter=',')[:, 1:]

    return connectionmatrix, variables


def read_scalelimits(scaling_loc):
    """Imports the scale limits for the data.
    The format of the CSV file should be:
    var, low, nominal, high, vartype (first row)
    var1, float, float, float, ['D', 'S'] (second row)
    var2, float, float, float, ['D, 'S'] (third row)
    etc...

    type 'D' indicates disturbance variable and maximum deviation will be used
    type 'S' indicates state variable and minimum deviation will be used

    """
    scalingdf = pd.read_csv(scaling_loc)
    scalingdf.set_index('var', inplace=True)

    return scalingdf


def read_biasvector(biasvector_loc):
    """Imports the bias vector for ranking purposes.
    The format of the CSV file should be:
    var1, var2, etc ... (first row)
    bias1, bias2, etc ... (second row)
    """
    with open(biasvector_loc) as f:
        variables = csv.reader(f).next()[1:]
        biasvector = np.genfromtxt(f, delimiter=',')[:]
    return biasvector, variables


def read_header_values_datafile(location):
    """This method reads a CSV data file of the form:
    header, header, header, etc... (first row)
    value, value, value, etc... (second row)
    etc...

    """

    with open(location) as f:
        header = next(csv.reader(f))[:]
        values = np.genfromtxt(f, delimiter=',')

    return values, header


def read_matrix(matrix_loc):
    """This method reads a matrix scheme for a specific scenario.

    Might need to pad matrix with zeros if it is non-square
    """
    with open(matrix_loc) as f:
        matrix = np.genfromtxt(f, delimiter=',')

    # Remove labels
    matrix = matrix[1:, 1:]

    return matrix


def buildcase(dummyweight, digraph, name, dummycreation):
    if dummycreation:
        counter = 1
        for node in digraph.nodes():
            if digraph.in_degree(node) == 1:
                # TODO: Investigate the effect of different weights
                nameofscale = name + str(counter)
                digraph.add_edge(nameofscale, node, weight=dummyweight)
                digraph.add_node(nameofscale, bias=1.)
                counter += 1

    connection = nx.to_numpy_matrix(digraph, weight=None)
    gain = nx.to_numpy_matrix(digraph, weight='weight')
    variablelist = digraph.nodes()
    nodedatalist = digraph.nodes(data=True)

    biaslist = []
    for node in digraph.nodes():
        biaslist.append(nodedatalist[node]['bias'])

    return np.array(connection), gain, variablelist, biaslist


def buildgraph(variables, gainmatrix, connections, biasvector):
    digraph = nx.DiGraph()
    # Construct the graph with connections
    for col, colvar in enumerate(variables):
        for row, rowvar in enumerate(variables):
            # The node order is source, sink according to
            # the convention that columns are sources and rows are sinks
            if connections[row, col] != 0:
                digraph.add_edge(rowvar, colvar, weight=gainmatrix[row, col])

    # Add the bias information to the graph nodes
    for nodeindex, nodename in enumerate(variables):
        digraph.add_node(nodename, bias=biasvector[nodeindex])

    return digraph


def write_dictionary(filename, dictionary):
    with open(filename, 'w') as f:
        json.dump(dictionary, f)


def rankbackward(variables, gainmatrix, connections, biasvector,
                 dummyweight, dummycreation):
    """This method adds a unit gain node to all nodes with an out-degree
    of 1 in order for the relative scale to be retained.
    Therefore all nodes with pointers should have 2 or more edges
    pointing away from them.

    It uses the number of dummy variables to construct these gain,
    connection and variable name matrices.

    """

    # TODO: Modify bias vector to assign zero weight to all dummy nodes

    digraph = buildgraph(variables, gainmatrix, connections, biasvector)
    return buildcase(dummyweight, digraph, 'DV BWD ', dummycreation)


def get_box_endates(clean_df, window, overlap, freq):
    """Gets the end dates of boxes from dataframe that are continous over window and guarenteed to have a maximum
    overlap.

    clean_df: clean dataframe with nan assigned to all bad data
    window: size of window in steps at desired frequency
    overlap: size of minium overlap desired in steps at desired frequency
    """

    # Calculate the minimum timedelta between start of boxes
    min_timedelta = pd.Timedelta(freq) * overlap

    clean_df = clean_df.resample(freq).mean()  # Resamples at desired frequency

    # Any aggregate function that returns nan when a nan occurs in window is suitable
    rolling_clean_df = clean_df.rolling(window=window, min_periods=window).mean()
    rolling_clean_df.dropna(inplace=True)  # All indexes that remain have window continous samples at freq

    index_diffs = (rolling_clean_df.index[1:] - rolling_clean_df.index[:-1])
    index_diffs_series = pd.Series(index_diffs, index=rolling_clean_df.index[1:])

    indexes_that_pass = index_diffs_series > pd.Timedelta(min_timedelta)

    bin_end_indexes = [rolling_clean_df.index[0]] + list(rolling_clean_df[1:][indexes_that_pass].index)

    return bin_end_indexes


def get_continous_boxes(clean_df, window, overlap, freq):

    box_end_dates = get_box_endates(clean_df, window, overlap, freq)
    boxdates = [
        np.asarray([(box_end_date - (pd.Timedelta(freq) * window)).value // 10 ** 9,
                    box_end_date.value // 10 ** 9])
        for box_end_date in box_end_dates]

    boxes = [clean_df[(box_end_date - (pd.Timedelta(freq) * (window - 1))):box_end_date]
             for box_end_date in box_end_dates]

    array_boxes = [np.asarray(box) for box in boxes]

    return array_boxes, boxdates


def split_tsdata(inputdata, samplerate, boxsize, boxnum):
    """Splits the inputdata into arrays useful for analysing the change of
    weights over time.

    inputdata is a numpy array containing values for a single variable
    (after sub-sampling)

    samplerate is the rate of sampling in time units (after sub-sampling)
    boxsize is the size of each returned dataset in time units
    boxnum is the number of boxes that need to be analyzed

    Boxes are evenly distributed over the provided dataset.
    The boxes will overlap if boxsize*boxnum is more than the simulated time,
    and will have spaces between them if it is less.


    """
    # Get total number of samples
    samples = len(inputdata)
#    print "Number of samples: ", samples
    # Convert boxsize to number of samples
    boxsizesamples = int(round(boxsize / samplerate))
#    print "Box size in samples: ", boxsizesamples
    # Calculate starting index for each box

    if boxnum == 1:
        boxes = [inputdata]

    else:
        boxstartindex = np.empty((1, boxnum))[0]
        boxstartindex[:] = np.NAN
        boxstartindex[0] = 0
        boxstartindex[-1] = samples - boxsizesamples
        samplesbetween = \
            ((float(samples - boxsizesamples)) / float(boxnum - 1))
        boxstartindex[1:-1] = [round(samplesbetween * index)
                               for index in range(1, boxnum-1)]
        boxes = [inputdata[int(boxstartindex[i]):int(boxstartindex[i]) +
                           int(boxsizesamples)]
                 for i in range(int(boxnum))]

    return boxes


def calc_signalent(vardata, weightcalcdata):
    """Calculates single signal differential entropies
    by making use of the JIDT continuous box-kernel implementation.

    """

    # Setup Java class for infodynamics toolkit
    entropyCalc, estimator = \
        transentropy.setup_infodynamics_entropy(weightcalcdata.infodynamicsloc)

    entropy = transentropy.calc_infodynamics_entropy(
        entropyCalc, vardata, estimator)
    return entropy


def vectorselection(data, timelag, sub_samples, k=1, l=1):
    """Generates sets of vectors from tags time series data
    for calculating transfer entropy.

    For notation references see Shu2013.

    Takes into account the time lag (number of samples between vectors of the
    same variable).

    In this application the prediction horizon (h) is set to equal
    to the time lag.

    The first vector in the data array should be the samples of the variable
    to be predicted (x) while the second vector should be sampled of the vector
    used to make the prediction (y).

    sub_samples is the amount of samples in the dataset used to calculate the
    transfer entropy between two vectors and must satisfy
    sub_samples <= samples

    The required number of samples is extracted from the end of the vector.
    If the vector is longer than the number of samples specified plus the
    desired time lag then the remained of the data will be discarded.

    k refers to the dimension of the historical data to be predicted (x)

    l refers to the dimension of the historical data used
    to do the prediction (y)

    """
    _, sample_n = data.shape
    x_pred = data[0, sample_n-sub_samples:]
    x_pred = x_pred[np.newaxis, :]

    x_hist = np.zeros((k, sub_samples))
    y_hist = np.zeros((l, sub_samples))

    for n in range(1, k+1):
        # Original form according to Bauer (2007)
        # TODO: Provide for comparison
        # Modified form according to Shu & Zhao (2013)
        startindex = (sample_n - sub_samples) - timelag*(n - 1) - 1
        endindex = sample_n - timelag*(n - 1) - 1
        x_hist[n-1, :] = data[1, startindex:endindex]
    for m in range(1, l+1):
        startindex = (sample_n - sub_samples) - timelag*m - 1
        endindex = sample_n - timelag*m - 1
        y_hist[m-1:, :] = data[0, startindex:endindex]

    return x_pred, x_hist, y_hist
