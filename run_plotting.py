"""Generates various plots as specified in the plotting configuration file."""

import json
import logging
import os

from faultmap import config_setup
from plotting.plotter import draw_plot

logging.basicConfig(level=logging.INFO)
dataloc, configloc, _, _ = config_setup.get_locations()

plotting_config = json.load(open(os.path.join(configloc, "config_plotting.json")))

# Flag indicating whether generated plots should be written to disk
writeoutput = plotting_config["writeoutput"]
# Provide the mode and case names to calculate
mode = plotting_config["mode"]
cases = plotting_config["cases"]

for case in cases:
    draw_plot(mode, case, writeoutput)
