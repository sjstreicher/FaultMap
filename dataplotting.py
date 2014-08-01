# -*- coding: utf-8 -*-
"""Plots the specified columns from a dataset

Created on Sat Apr 05 08:19:51 2014

@author: Simon Streicher
"""

import logging
import os
import json
from sklearn import preprocessing
import numpy as np
import h5py
import matplotlib.pyplot as plt

from config_setup import runsetup
from ranking.formatmatrices import read_connectionmatrix
from config_setup import ensure_existance

# Define the mode and case for plot generation
mode = 'plants'
case = 'propylene_compressor'

#Specify whether the data should be normalised before plotting
normalise = True

saveloc, casedir, infodynamicsloc = runsetup(mode, case)
# Load case config file
caseconfig = json.load(open(os.path.join(casedir, case + '_fftplotting.json')))

# Get scenarios
scenarios = caseconfig['scenarios']
# Get data type
datatype = caseconfig['datatype']
sampling_rate = caseconfig['sampling_rate']

for scenario in scenarios:
    logging.info("Running scenario {}".format(scenario))

    if datatype == 'file':
        # Get time series data
        tags_tsdata = os.path.join(casedir, 'data',
                                   caseconfig[scenario]['data'])
        # Get connection (adjacency) matrix
        connectionloc = os.path.join(casedir, 'connections',
                                     caseconfig[scenario]['connections'])
        # Get dataset name
        dataset = caseconfig[scenario]['dataset']
        # Get inputdata
        inputdata = np.array(h5py.File(tags_tsdata, 'r')[dataset])
        # Get the variables and connection matrix
        [variables, connectionmatrix] = \
            read_connectionmatrix(connectionloc)

    elif datatype == 'function':
        tags_tsdata_gen = caseconfig[scenario]['datagen']
        connectionloc = caseconfig[scenario]['connections']
        # TODO: Store function arguments from config_setup import
        # ensure_existence scenario config file
        samples = caseconfig['gensamples']
        delay = caseconfig['delay']
        # Get inputdata
        inputdata = eval(tags_tsdata_gen)(samples, delay)
        # Get the variables and connection matrix
        [variables, connectionmatrix] = eval(connectionloc)()

    # Normalise (mean centre and variance scale) the input data

    if normalise is True:
        inputdata_norm = preprocessing.scale(inputdata, axis=0)
    else:
        inputdata_norm = inputdata

    def bandgap(min_freq, max_freq, vardata):
        """Bandgap filter based on FFT"""
        freqlist = np.fft.rfftfreq(vardata.size, sampling_rate)
        # Investigate effect of using abs()
        var_fft = np.fft.rfft(vardata)
        cut_var_fft = var_fft.copy()
        cut_var_fft[(freqlist < min_freq)] = 0
        cut_var_fft[(freqlist > max_freq)] = 0

        cut_vardata = np.fft.irfft(cut_var_fft)

        return cut_vardata

    for variable in variables:
        logging.info("Analysing effect of: " + variable)
#        vardata = inputdata_norm[:, variables.index(variable)]
        vardata = inputdata_norm[:, variables.index(variable)]

        # Make the directory to store the FFT plot if not already created
        plotdir = ensure_existance(os.path.join(saveloc, 'plots'), make=True)

        # Filter time series data
        vardata = bandgap(0.005, 0.008, vardata)

        # Create and save FFT plot
        # Compute FFT (normalised amplitude)
        var_fft = abs(np.fft.rfft(vardata)) * \
            (2. / len(vardata))
        freqlist = np.fft.rfftfreq(len(vardata), sampling_rate)

        # Select up to which frequency to plot
        fft_endsample = 500

        plt.figure()
        plt.plot(freqlist[0:fft_endsample], var_fft[0:fft_endsample],
                 'r', label=variable)
        plt.xlabel('Frequency (1/second)')
        plt.ylabel('Normalised amplitude')
        plt.legend()

        varmaxindex = var_fft.tolist().index(max(var_fft))
        print variable + " maximum signal strenght frequency: " + \
            str(freqlist[varmaxindex])

        filename_template = os.path.join(plotdir,
                                         'FFT_{}_{}_{}.pdf')

        def filename(variablename):
            return filename_template.format(case, scenario, variablename)

        plt.savefig(filename(variable))
#        plt.show()
        plt.close()
