"""
Calculates and writes weight_array and delay_array for different weight types
from data generated by run_weightcalc process.

@author: Simon Streicher
"""
import json
import logging
import os

import config_setup
from ranking.data_processing import result_reconstruction

logging.basicConfig(level=logging.INFO)

dataloc, configloc, _ = config_setup.get_locations()
createarrays_config = json.load(open(os.path.join(configloc, 'config'
                                                  '_createarrays' + '.json')))

writeoutput = createarrays_config['writeoutput']
mode = createarrays_config['mode']
cases = createarrays_config['cases']

for case in cases:
    result_reconstruction(mode, case, writeoutput)
