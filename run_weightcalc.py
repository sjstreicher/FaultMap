# -*- coding: utf-8 -*-
"""
@author: Simon Streicher

Performs weight calculation for all cases specified in the configuration file.

"""
# Standard modules
import json
import logging
import multiprocessing
import os

import config_setup
from ranking.gaincalc import weightcalc

if __name__ == '__main__':
    multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO)
    dataloc, configloc, _ = config_setup.get_locations()
    weightcalc_config = json.load(open(
        os.path.join(configloc, 'config_weightcalc' + '.json')))

    # Flag indicating whether calculated results should be written to disk
    writeoutput = weightcalc_config['writeoutput']
    # Flag indicating whether single signal entropy values for each
    # signal involved should be calculated
    single_entropies = weightcalc_config['calc_single_entropies']
    # Provide the mode and case names to calculate
    mode = weightcalc_config['mode']
    cases = weightcalc_config['cases']
    fftcalc = weightcalc_config['fft_calc']
    do_multiprocessing = weightcalc_config['multiprocessing']

    for case in cases:
        weightcalc(mode, case, writeoutput, single_entropies, fftcalc,
                   do_multiprocessing)
