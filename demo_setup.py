# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 21:00:03 2014

@author: Simon
"""

import json
import os


def runsetup(mode='test_cases', case='weightcalc_tests'):
    """Gets all required parameters from the case configuration file.

    Mode can be either 'test_cases' or 'plants' as required to get to the
    correct directory.

    """

    if mode == 'test_cases':
#        saveloc = os.path.join(os.path.dirname(__file__), 'texports')
        saveloc = os.path.join('test_exports')
#        casedir = os.path.join(os.path.dirname(__file__), 'tconfigs', case)
        casedir = os.path.join('test_configs')
        infodynamicsloc = 'infodynamics.jar'

    elif mode == 'plants':

        # Load directories config file
        dirs = json.load(open('config.json'))
        # Get data and preferred export directories from
        # directories config file
        dataloc = os.path.expanduser(dirs['dataloc'])
        saveloc = os.path.expanduser(dirs['saveloc'])
        infodynamicsloc = os.path.expanduser(dirs['infodynamicsloc'])
        # Define case data directory
        casedir = os.path.join(dataloc, mode, case)

    return saveloc, casedir, infodynamicsloc
