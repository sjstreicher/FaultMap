# -*- coding: utf-8 -*-
"""
Demonstrates Skogestad scaling
"""

import os
import sys
import pandas as pd
import numpy as np
import h5py
import sklearn

sys.path.append(os.getcwd())
from ranking import data_processing

scaling_loc = 'todo/scaling_example_limits.csv'


raw_tsdata = 'todo/scaling_example_data.csv'
saveloc = 'C:/Repos/FaultMap/todo'
scenario = 'test'
casename = 'test'
datapath = data_processing.csv_to_h5(saveloc, raw_tsdata, scenario, casename)

variables = data_processing.read_variables(raw_tsdata)

inputdata_raw = np.array(h5py.File(os.path.join(
                         datapath, scenario + '.h5'), 'r')[scenario])

scalingvalues = data_processing.read_scalelimits(scaling_loc)

inputdata_raw = np.array(h5py.File(os.path.join(
                datapath, scenario + '.h5'), 'r')[scenario])


def scale_select(vartype, lower_limit, nominal_level, high_limit):
    if vartype == 'D':
        limit = max((nominal_level - lower_limit),
                    (high_limit - nominal_level))
    elif vartype == 'S':
        limit = min((nominal_level - lower_limit),
                    (high_limit - nominal_level))
    else:
        raise NameError("Variable type flag not recognized")
    return limit


def skogestad_scale(data_raw, variables, scalingvalues):
    if scalingvalues is None:
        raise ValueError("Scaling values not defined")

    data_skogestadscaled = np.zeros_like(data_raw)

    scalingvalues['scale_factor'] = map(
        scale_select, scalingvalues['vartype'], scalingvalues['low'],
        scalingvalues['nominal'], scalingvalues['high'])

    # Loop through variables
    # The variables are aligned with the columns in raw_data
    for index, var in enumerate(variables):
        factor = scalingvalues.loc[var]['scale_factor']
        nominalval = scalingvalues.loc[var]['nominal']
        data_skogestadscaled[:, index] = \
            (data_raw[:, index] - nominalval) / factor

    return data_skogestadscaled

inputdata_skogestad_scaled = \
    skogestad_scale(inputdata_raw, variables, scalingvalues)

inputdata_standardised = \
    sklearn.preprocessing.scale(inputdata_raw, axis=0)
