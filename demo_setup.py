# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 21:00:03 2014

@author: Simon
"""

import json
import os


def runsetup(mode='test_cases'):
    # Load directories config file
    dirs = json.load(open('config.json'))
    # Get data and preferred export directories from directories config file
    dataloc = os.path.expanduser(dirs['dataloc'])
    saveloc = os.path.expanduser(dirs['saveloc'])
    infodynamicsloc = os.path.expanduser(dirs['infodynamicsloc'])

    # Define plant and or test case name to run
    case = 'autoreg_2x2'
    # Define plant data directory
    casedir = os.path.join(dataloc, mode, case)
    scenarios = ['fullconn', 'correctconn', 'reverseconn']
    # Load scenario config file
    scenconfig = json.load(open(os.path.join(casedir, case + '.json')))
    # Get sampling rate
    sampling_rate = scenconfig['sampling_rate']

    return scenarios, saveloc, scenconfig, casedir, sampling_rate, \
        infodynamicsloc
