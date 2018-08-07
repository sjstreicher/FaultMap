# -*- coding: utf-8 -*-
"""Performs node ranking.

"""
import json
import logging
import os

import config_setup
from ranking.noderank import noderankcalc

logging.basicConfig(level=logging.INFO)

dataloc, configloc, saveloc, _ = config_setup.get_locations()
noderank_config = json.load(open(os.path.join(configloc, "config_noderank" + ".json")))

writeoutput = noderank_config["writeoutput"]
mode = noderank_config["mode"]
cases = noderank_config["cases"]

for case in cases:
    noderankcalc(mode, case, writeoutput)
