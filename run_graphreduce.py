# -*- coding: utf-8 -*-

"""
@author: Simon Streicher

"""
import logging
import json
import os

import config_setup
from ranking.graphreduce import reducegraph

logging.basicConfig(level=logging.INFO)

dataloc, _ = config_setup.get_locations()
graphreduce_config = json.load(open(os.path.join(dataloc, 'config'
                                                 '_graphreduce' + '.json')))

writeoutput = graphreduce_config['writeoutput']
mode = graphreduce_config['mode']
cases = graphreduce_config['cases']

for case in cases:
    reducegraph(mode, case, writeoutput)
