# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 21:00:03 2014

@author: Simon
"""

import json
import os


def runsetup(mode='test_cases', case='delaytests'):
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
    # Load case config file
    caseconfig = json.load(open(os.path.join(casedir, case + '.json')))

    return saveloc, caseconfig, casedir, infodynamicsloc
