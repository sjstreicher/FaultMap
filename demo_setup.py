# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 21:00:03 2014

@author: Simon
"""

import json
import os


def runsetup(mode='test_cases', case='random_2x2'):
    """Gets all required parameters from the case configuration file.

    Mode can be either 'test_cases' or 'plants' as required to get to the
    correct directory.

    """

    # Load directories config file
    dirs = json.load(open('config.json'))
    # Get data and preferred export directories from directories config file
    dataloc = os.path.expanduser(dirs['dataloc'])
    saveloc = os.path.expanduser(dirs['saveloc'])
    infodynamicsloc = os.path.expanduser(dirs['infodynamicsloc'])

    # Define case data directory
    casedir = os.path.join(dataloc, mode, case)
    # Load scenario config file
    scenconfig = json.load(open(os.path.join(casedir, case + '.json')))
    # Get scenarios
    scenarios = scenconfig['scenarios']
    # Get sampling rate
    sampling_rate = scenconfig['sampling_rate']
    # Get data type
    datatype = scenconfig['datatype']

    return scenarios, saveloc, scenconfig, casedir, sampling_rate, \
        infodynamicsloc, datatype
