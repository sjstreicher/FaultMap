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
case = 'epu5_compressor'

# Amount of samples to lag cause variable behind affected variable
delay = 0

#Specify whether the data should be normalised before plotting
normalise = True

saveloc, casedir, infodynamicsloc = runsetup(mode, case)
# Load case config file
caseconfig = json.load(open(os.path.join(casedir, case + '.json')))

# Get scenarios
scenarios = caseconfig['scenarios']
# Get data type
datatype = caseconfig['datatype']
testsize = caseconfig['testsize']

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
        # TODO: Store function arguments ifrom config_setup import ensure_existancen scenario config file
        samples = caseconfig['gensamples']
        delay = caseconfig['delay']
        # Get inputdata
        inputdata = eval(tags_tsdata_gen)(samples, delay)
        # Get the variables and connection matrix
        [variables, connectionmatrix] = eval(connectionloc)()

    # Normalise (mean centre and variance scale) the input data
    inputdata = inputdata[5200:5300]
    inputdata_norm = preprocessing.scale(inputdata, axis=0)
#    inputdata_norm = inputdata

    for causevarindex in [31, 32]:
        causevar = variables[causevarindex]
        logging.info("Analysing effect of: " + causevar)
        for affectedvarindex in [11]:
            affectedvar = variables[affectedvarindex]
            if not(connectionmatrix[affectedvarindex, causevarindex] == 0):

                if delay == 0:
                    offset = None
                else:
                    offset = -delay

                causevardata = \
                    inputdata_norm[:, causevarindex]
                affectedvardata = \
                    inputdata_norm[:, affectedvarindex]

                time = range(len(causevardata))

                plt.figure(1)
                plt.plot(time, causevardata, 'b', label=causevar)
                plt.hold(True)
                plt.plot(time, affectedvardata, 'r', label=affectedvar)
                plt.xlabel('Time (minutes)')
                plt.ylabel('Normalised value')
                plt.legend()


                plotdir = ensure_existance(os.path.join(saveloc, 'plots'),
                                           make=True)

                filename_template = os.path.join(plotdir, '{}_{}_{}_{}.pdf')

                def filename(causename, affectedname):
                    return filename_template.format(case, scenario,
                                                    causename, affectedname)

                plt.savefig(filename(causevar, affectedvar))

                plt.clf()