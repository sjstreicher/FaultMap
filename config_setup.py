# -*- coding: utf-8 -*-
"""
Setup functions used to read configuration files.

@author: Simon Streicher
"""

import json
import os


def ensure_existence(location, make=True):
    if not os.path.exists(location):
        if make:
            os.makedirs(location)
        else:
            raise IOError("File does not exists: {}".format(location))
    return location


def runsetup(mode, case):
    """Gets all required directories from the case configuration file.

    Parameters
    ----------
        mode : string
            Either 'tests' or 'cases'. Tests data are generated dynamically and
            stored in specified folders. Case data are read from file and stored
            under organized headings in the saveloc directory specified
            in config.json.
        case : string
            The name of the case that is to be run. Points to dictionary in either
            test or case config files.

    Returns
    -------
        saveloc : path
        caseconfigloc : path
        casedir : path
        infodynamicsloc : path

    """

    if mode == 'tests':
        saveloc = 'test_exports'
        casedir = 'test_configs'
        caseconfigdir = 'test_configs'
        infodynamicsloc = 'infodynamics.jar'

    elif mode == 'cases':

        # Load directories config file
        dirs = json.load(open('config.json'))
        # Get data and preferred export directories from
        # directories config file
        locations = [ensure_existence(os.path.expanduser(dirs[location]))
                     for location in ['dataloc', 'configloc', 'saveloc',
                                      'infodynamicsloc']]
        dataloc, configloc, saveloc, infodynamicsloc = locations

        # Define case data directory
        casedir = ensure_existence(os.path.join(dataloc, mode, case),
                                   make=True)
        caseconfigdir = os.path.join(configloc, mode, case)

    return saveloc, caseconfigdir, casedir, infodynamicsloc


def get_locations():
    # Load directories config file
    dirs = json.load(open('config.json'))
    dataloc, configloc, saveloc = \
        [os.path.expanduser(dirs[location])
         for location in ['dataloc', 'configloc', 'saveloc']]
    return dataloc, configloc, saveloc
