# -*- coding: utf-8 -*-
"""
@author: Simon Streicher

This module is used to generate various plots as specified in the
plotting configuration file.

"""

import logging
import json
import os


import config_setup
from plotting.plotter import plotdraw

logging.basicConfig(level=logging.INFO)
dataloc, configloc, _ = config_setup.get_locations()

plotting_config = json.load(open(
    os.path.join(configloc, 'config_plotting' + '.json')))

# Flag indicating whether generated plots should be written to disk
writeoutput = plotting_config['writeoutput']
# Provide the mode and case names to calculate
mode = plotting_config['mode']
cases = plotting_config['cases']

for case in cases:
    plotdraw(mode, case, writeoutput)
