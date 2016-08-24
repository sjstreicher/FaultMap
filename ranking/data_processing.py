"""Performs various data processings support tasks called by the gaincalc
module.

@author: St. Elmo Wilken, Simon Streicher

"""

import numpy as np
import tables as tb
import networkx as nx
import pandas as pd
import csv
import sklearn.preprocessing
import os
import matplotlib.pyplot as plt
import json

# Own libraries
import config_setup
import transentropy

import gaincalc


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


def UnivariateSurrogates(data_f, MaxIter):
    """Generates iAAFT surrogates
    
    """
    
    xs=data_f.copy()
    xs.sort() #sorted amplitude stored
    pwx=np.abs(np.fft.fft(data_f)) # amplitude of fourier transform of orig
    
    data_f.shape = (-1,1)
    xsur = np.random.permutation(data_f) #random permutation as starting point
    xsur.shape = (1,-1)
    
    for i in range(MaxIter):
        fftsurx = pwx*np.exp(1j*np.angle(np.fft.fft(xsur)))
        xoutb = np.real(np.fft.ifft(fftsurx))
        ranks = xoutb.argsort(axis=1)
        xsur[:,ranks] = xs
        
    return(xsur) 

def process_auxfile(filename, allow_neg=False):
    """Processes an auxfile and returns a list of affected_vars,
    weight_array as well as relative significance weight array.

    """

    affectedvars = []
    weights = []
    nosigtest_weights = []
    sigweights = []
    delays = []

    with open(filename, 'rb') as auxfile:
        auxfilereader = csv.reader(auxfile, delimiter=',')
        for rowindex, row in enumerate(auxfilereader):
            if rowindex == 0:
                # Find the indices of important rows
                affectedvar_index = row.index('affectedvar')

                if 'max_ent' in row:
                    maxval_index = row.index('max_ent')
                else:
                    maxval_index = row.index('max_corr')

                if 'threshold' in row:
                    thresh_index = row.index('threshold')
                else:
                    thresh_index = row.index('threshcorr')

                threshpass_index = row.index('threshpass')
                maxdelay_index = row.index('max_delay')

            if rowindex > 0:

                affectedvars.append(row[affectedvar_index])

                # Test if weight failed threshpass test and write as zero
                # if true

                # In rare cases it might be desired to allow negative values
                # (e.g. correlation tests)
                # TODO: Put the allow_neg parameter in a configuration file
                # NOTE: allow_neg also removes significance testing
                weight_candidate = float(row[maxval_index])

                if allow_neg:
                    nosigtest_weights.append(weight_candidate)
                    weights.append(weight_candidate)
                else:
                    if weight_candidate > 0.:
                        # Attach to no significance test result
                        nosigtest_weights.append(weight_candidate)
                        if row[threshpass_index] == 'False':
                            weights.append(0.)
                        else:
                            # threshpass is either None or True
                            weights.append(weight_candidate)
                    else:
                        weights.append(0.)
                        nosigtest_weights.append(0.)

                delays.append(float(row[maxdelay_index]))

                # Test if sigtest passed before assigning weight
                if row[threshpass_index] == 'True':
                    # If the threshold is negative, take the absolute value
                    # TODO: Need to think the implications of this through
                    threshold = float(row[thresh_index])
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

    return affectedvars, weights, nosigtest_weights, sigweights, delays


def create_arrays(datadir, variables):
    """
    datadir is the location of the auxdata and weights folders for the
    specific case that is under investigation

    tsfilename is the file name of the original time series data file
    used to generate each case and is only used for generating a list of
    variables

    """

    absoluteweightarray_name = 'weight_absolute_arrays'
    directionalweightarray_name = 'weight_directional_arrays'
    neutralweightarray_name = 'weight_arrays'
    absolutesigweightarray_name = 'sigweight_absolute_arrays'
    directionalsigweightarray_name = 'sigweight_directional_arrays'
    neutralsigweightarray_name = 'sigweight_arrays'
    absolutedelayarray_name = 'delay_absolute_arrays'
    directionaldelayarray_name = 'delay_directional_arrays'
    neutraldelayarray_name = 'delay_arrays'

    directories = next(os.walk(datadir))[1]

    test_strings = ['auxdata_absolute', 'auxdata_directional', 'auxdata']

    for test_string in test_strings:

        if test_string in directories:

            if test_string == 'auxdata_absolute':
                weightarray_name = absoluteweightarray_name
                sigweightarray_name = absolutesigweightarray_name
                delayarray_name = absolutedelayarray_name
            elif test_string == 'auxdata_directional':
                weightarray_name = directionalweightarray_name
                sigweightarray_name = directionalsigweightarray_name
                delayarray_name = directionaldelayarray_name
            elif test_string == 'auxdata':
                weightarray_name = neutralweightarray_name
                sigweightarray_name = neutralsigweightarray_name
                delayarray_name = neutraldelayarray_name

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
                for causevar_file in causevar_filenames:
                    causevars.append(str(causevar_file[:-4]))

                    # Open auxfile and return weight array as well as
                    # significance relative weight arrays

                    (affectedvars, weights, nosigtest_weights,
                     sigweights, delays) = \
                        process_auxfile(os.path.join(boxdir, causevar_file))

                    affectedvar_array.append(affectedvars)
                    weight_array.append(weights)
                    nosigtest_weight_array.append(nosigtest_weights)
                    sigweight_array.append(sigweights)
                    delay_array.append(delays)

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

                nosigtest_dirparts = getfolders(datadir)
                if 'sigtested' in nosigtest_dirparts:

                    nosigtest_dirparts[
                        nosigtest_dirparts.index('sigtested')] = 'nosigtest'
                    nosigtest_savedir = nosigtest_dirparts[0]
                    for pathpart in nosigtest_dirparts[1:]:
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

    return None


def create_signtested_directionalarrays(datadir, writeoutput):
    """
    Checks whether the directional weight arrays have corresponding
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
                 'delay_arrays': 'delay_trend'}

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
                    'delay_arrays': 'delay_array'}

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

    The results are written to the same folders where the files are found.


    """

    saveloc, caseconfigdir, casedir, _ = config_setup.runsetup(mode, case)

    caseconfig = json.load(
        open(os.path.join(caseconfigdir, case + '_weightcalc' + '.json')))

    # Directory where subdirectories for scenarios will be stored
    scenariosdir = os.path.join(saveloc, 'weightdata', case)

    # Get list of all scenarios
    scenarios = next(os.walk(scenariosdir))[1]

    for scenario in scenarios:
        print scenario

        weightcalcdata = \
            gaincalc.WeightcalcData(mode, case, False, False, False)
        weightcalcdata.setsettings(scenario,
                                   caseconfig[scenario]['settings'][0])

        methodsdir = os.path.join(scenariosdir, scenario)
        methods = next(os.walk(methodsdir))[1]
        for method in methods:
            print method
            sigtypesdir = os.path.join(methodsdir, method)
            sigtypes = next(os.walk(sigtypesdir))[1]
            for sigtype in sigtypes:
                print sigtype
                embedtypesdir = os.path.join(sigtypesdir, sigtype)
                embedtypes = next(os.walk(embedtypesdir))[1]
                for embedtype in embedtypes:
                    print embedtype
                    datadir = os.path.join(embedtypesdir, embedtype)
                    create_arrays(datadir, weightcalcdata.variables)
                    # Provide directional array version tested with absolute
                    # weight sign
                    create_signtested_directionalarrays(datadir, writeoutput)

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
        print scenario

        methodsdir = os.path.join(scenariosdir, scenario)
        methods = next(os.walk(methodsdir))[1]
        for method in methods:
            print method
            sigtypesdir = os.path.join(methodsdir, method)
            sigtypes = next(os.walk(sigtypesdir))[1]
            for sigtype in sigtypes:
                print sigtype
                embedtypesdir = os.path.join(sigtypesdir, sigtype)
                embedtypes = next(os.walk(embedtypesdir))[1]
                for embedtype in embedtypes:
                    print embedtype
                    datadir = os.path.join(embedtypesdir, embedtype)
                    extract_trends(datadir, writeoutput)

    return None


def csv_to_h5(saveloc, raw_tsdata, scenario, case):

    # Name the dataset according to the scenario
    dataset = scenario

    datapath = config_setup.ensure_existence(os.path.join(
        saveloc, 'data', case), make=True)

    filename = os.path.join(datapath, scenario + '.h5')

    if not os.path.exists(filename):

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
    return timestamps


def read_variables(raw_tsdata):
    with open(raw_tsdata) as f:
        variables = csv.reader(f).next()[1:]
    return variables


def writecsv(filename, items, header=None):
    """Write CSV directly"""
    with open(filename, 'wb') as f:
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
    print "Starting FFT calculations"

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

#    logging.info("Done with FFT calculations")
    print "Done with FFT calculations"

    return None


def bandgap(min_freq, max_freq, vardata):
    """Bandgap filter based on FFT"""
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


def descriptive_dictionary(descriptive_file):
    """Converts the description CSV file to a dictionary."""
    descriptive_array = np.genfromtxt(descriptive_file, delimiter=',',
                                      dtype='string')
    tag_names = descriptive_array[1:, 0]
    tag_descriptions = descriptive_array[1:, 1]
    description_dict = dict(zip(tag_names, tag_descriptions))
    return description_dict


def normalise_data(raw_tsdata, inputdata_raw, saveloc, case, scenario):
    # Header and time from main source file
    headerline = np.genfromtxt(raw_tsdata, delimiter=',', dtype='string')[0, :]
    time = np.genfromtxt(raw_tsdata, delimiter=',')[1:, 0]
    time = time[:, np.newaxis]

    inputdata_normalised = \
        sklearn.preprocessing.scale(inputdata_raw, axis=0)

    datalines = np.concatenate((time, inputdata_normalised), axis=1)

    # Define export directories and filenames
    datadir = config_setup.ensure_existence(
        os.path.join(saveloc, 'normdata'), make=True)

    filename_template = os.path.join(datadir, '{}_{}_{}.csv')

    def filename(name):
        return filename_template.format(case, scenario, name)

    # Store the normalised data in similar format as original data
    writecsv(filename('normalised_data'), datalines, headerline)

    return inputdata_normalised


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

    Value is 1 if column variable points to row variable
    (causal relationship)
    Value is 0 otherwise

    """
    with open(connection_loc) as f:
        variables = csv.reader(f).next()[1:]
        connectionmatrix = np.genfromtxt(f, delimiter=',')[:, 1:]

    return connectionmatrix, variables


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
        header = csv.reader(f).next()[:]
        values = np.genfromtxt(f, delimiter=',')

    return values, header


def read_matrix(matrix_loc):
    """This method a matrix scheme for a specific scenario.

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
    for nodeindex, node in enumerate(digraph.nodes()):
        biaslist.append(nodedatalist[nodeindex][1]['bias'])

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
    with open(filename, 'wb') as f:
        json.dump(dictionary, f)


def read_dictionary(filename):
    with open(filename, 'rb') as f:
        return json.load(f)


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
#        samplesbetween = int(round((samples - boxsizesamples) / boxnum))
        samplesbetween = \
            ((float(samples) - float(boxsizesamples)) / float(boxnum))
        boxstartindex[1:-1] = [round(samplesbetween * index)
                               for index in range(1, boxnum-1)]
        boxes = [inputdata[int(boxstartindex[i]):int(boxstartindex[i]) +
                           int(boxsizesamples)]
                 for i in range(int(boxnum))]
    return boxes


def ewma_weights_benchmark(weights, alpha_rate):
    """Calculates an exponential moving average of weights
    for different boxes to use as a benchmark.

    weights is a list of weights for different boxes

    """
    benchmark_weights = np.zeros_like(weights)

    for index, weight in enumerate(weights):
        if index == 0:
            benchmark_weights[index] = weights[index]
        else:
            benchmark_weights[index] = \
                (alpha_rate * benchmark_weights[index-1]) + \
                ((1-alpha_rate) * weight)

    return benchmark_weights


def calc_signalent(vardata, weightcalcdata):
    """Calculates single signal differential entropies
    by making use of the JIDT continuous box-kernel implementation.

    """

    # Setup Java class for infodynamics toolkit
    entropyCalc = \
        transentropy.setup_infodynamics_entropy(weightcalcdata.normalize)

    entropy = transentropy.calc_infodynamics_entropy(entropyCalc, vardata.T)
    return entropy
