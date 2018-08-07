# -*- coding: utf-8 -*-
"""Setup functions used to read configuration files.

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


def get_locations(mode="cases"):
    """Gets all required directories related to the specified mode.

    Parameters
    ----------
        mode : string
            Either 'tests' or 'cases'. Specifies whether the test or user
            configureable cases directories should be set.
            Test directiories are read from testconfig.json which is bundled
            with the code, while cases directories are read from
            caseconfig.json which must be created by the user.

    Returns
    -------
        dataloc : path
        configloc : path
        saveloc : path
        infodynamicsloc : path

    """
    # Load directories config file
    if mode == "tests":
        dirs = json.load(open("testconfig.json"))
    elif mode == "cases":
        dirs = json.load(open("caseconfig.json"))
    else:
        raise NameError("Mode name not recognized")

    # Get data and preferred export directories from
    # directories config file
    locations = [
        ensure_existence(os.path.expanduser(dirs[location]))
        for location in ["dataloc", "configloc", "saveloc", "infodynamicsloc"]
    ]
    dataloc, configloc, saveloc, infodynamicsloc = locations

    return dataloc, configloc, saveloc, infodynamicsloc


def runsetup(mode, case):
    """Gets all required directories from the case configuration file.

    Parameters
    ----------
        mode : string
            Either 'tests' or 'cases'. Specifies whether the test or user
            configureable cases directories should be set.
            Test directiories are read from testconfig.json which is bundled
            with the code, while cases directories are read from
            caseconfig.json which must be created by the user.
        case : string
            The name of the case that is to be run. Points to dictionary
            in either test or case config files.

    Returns
    -------
        saveloc : path
        caseconfigdir : path
        casedir : path
        infodynamicsloc : path

    """

    dataloc, configloc, saveloc, infodynamicsloc = get_locations(mode)

    # Define case data directory
    casedir = ensure_existence(os.path.join(dataloc, mode, case), make=True)
    caseconfigdir = os.path.join(configloc, mode, case)

    return saveloc, caseconfigdir, casedir, infodynamicsloc
