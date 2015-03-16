"""Performs various data processings support tasks called by the gaincalc
module.

@author: St. Elmo Wilken, Simon Streicher

"""

import numpy as np
import tables as tb
import networkx as nx
import csv
import sklearn.preprocessing
import os
import matplotlib.pyplot as plt
import json
import logging

import config_setup

# Own libraries
import transentropy


def csv_to_h5(saveloc, raw_tsdata, scenario, case):

    # Name the dataset according to the scenario
    dataset = scenario

    datapath = config_setup.ensure_existance(os.path.join(
        saveloc, 'data', case), make=True)

    hdf5writer = tb.open_file(os.path.join(datapath, scenario + '.h5'), 'w')
    data = np.genfromtxt(raw_tsdata, delimiter=',')
    # Strip time column and labels first row
    data = data[1:, 1:]
    array = hdf5writer.create_array(hdf5writer.root, dataset, data)

    array.flush()
    hdf5writer.close()

    return datapath


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


def fft_calculation(raw_tsdata, normalised_tsdata, variables, sampling_rate,
                    sampling_unit, saveloc, case, scenario,
                    plotting=False, plotting_endsample=500):

    # logging.info("Starting FFT calculations")
    # Using a print command instead as logging is late
    print "Starting FFT calculations"

    headerline = np.genfromtxt(raw_tsdata, delimiter=',', dtype='string')[0, :]

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

            plotdir = config_setup.ensure_existance(
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
    datadir = config_setup.ensure_existance(
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
    datadir = config_setup.ensure_existance(
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
    datadir = config_setup.ensure_existance(
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


def read_gainmatrix(gainmatrix_loc):
    """This method a gainmatrix scheme for a specific scenario.

    Might need to pad gainmatrix with zeros if it is non-square
    """
    with open(gainmatrix_loc) as f:
        gainmatrix = np.genfromtxt(f, delimiter=',')

    return gainmatrix


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
            if (connections[row, col] != 0):
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
    and will have spaced between them if it is less.


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
        samplesbetween = int(round(samples/boxnum))
        boxstartindex[1:-1] = [(samplesbetween * index)
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
                ((1-alpha_rate) * weights[index])

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
