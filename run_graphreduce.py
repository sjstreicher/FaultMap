"""Performs graph reduction."""

import json
import logging
import os

from faultmap import config_setup
from faultmap.graphreduce import reduce_graph_scenarios

logging.basicConfig(level=logging.INFO)

dataloc, configloc, _, _ = config_setup.get_locations()
graphreduce_config = json.load(open(os.path.join(configloc, "config_graphreduce.json")))

writeoutput = graphreduce_config["writeoutput"]
mode = graphreduce_config["mode"]
cases = graphreduce_config["cases"]

for case in cases:
    reduce_graph_scenarios(mode, case, writeoutput)
