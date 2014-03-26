# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 21:00:03 2014

@author: Simon
"""

import json
import os


def runsetup(mode='plants'):
    """Mode can be either 'test_cases' or 'plants'"""
    # Load directories config file
    dirs = json.load(open('config.json'))
    # Get data and preferred export directories from directories config file
    dataloc = os.path.expanduser(dirs['dataloc'])
    saveloc = os.path.expanduser(dirs['saveloc'])
    infodynamicsloc = os.path.expanduser(dirs['infodynamicsloc'])

    # Define plant or test case name to run
    case = 'tennessee_eastman'
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
