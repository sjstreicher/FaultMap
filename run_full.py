# -*- coding: utf-8 -*-
"""
@author: Simon Streicher

Performs all analysis steps for all cases specified in the configuration file.

"""
# Standard modules
import json
import logging
import multiprocessing
import os

import config_setup
from ranking.gaincalc import weightcalc
from ranking.data_processing import result_reconstruction
from ranking.data_processing import trend_extraction
from ranking.noderank import noderankcalc
from plotting.plotter import plotdraw


def run_weightcalc(configloc, writeoutput, mode, case):
    weightcalc_config = json.load(open(
        os.path.join(configloc, 'config_weightcalc' + '.json')))

    # Flag indicating whether single signal entropy values for each
    # signal involved should be calculated
    single_entropies = weightcalc_config['calc_single_entropies']
    # Flag indicating whether
    fftcalc = weightcalc_config['fft_calc']
    do_multiprocessing = weightcalc_config['multiprocessing']

    try:
        weightcalc(mode, case, writeoutput, single_entropies, fftcalc,
                   do_multiprocessing)
    except:
        raise RuntimeError("Weight calculation failed for case: " + case)

    return None


def run_createarrays(writeoutput, mode, case):

    try:
        result_reconstruction(mode, case, writeoutput)
    except:
        raise RuntimeError("Array creation failed for case: " + case)

    return None


def run_trendextraction(writeoutput, mode, case):

    try:
        trend_extraction(mode, case, writeoutput)
    except:
        raise RuntimeError("Trend extraction failed for case: " + case)

    return None


def run_noderank(writeoutput, mode, case):

    try:
        noderankcalc(mode, case, writeoutput)
    except:
        raise RuntimeError("Node ranking failed for case: " + case)

    return None


def run_plotting(writeoutput, mode, case):

    try:
        plotdraw(mode, case, writeoutput)
    except:
        raise RuntimeError("Plotting failed for case: " + case)

    return None

if __name__ == '__main__':
    multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO)

    _, configloc, _ = config_setup.get_locations()
    fullrun_config = json.load(open(
        os.path.join(configloc, 'config_full' + '.json')))

    # Flag indicating whether calculated results should be written to disk
    writeoutput = fullrun_config['writeoutput']
    # Provide the mode and case names to calculate
    mode = fullrun_config['mode']
    cases = fullrun_config['cases']

    for case in cases:
        logging.info("Now attempting case: " + case)
        run_weightcalc(configloc, writeoutput, mode, case)
        run_createarrays(writeoutput, mode, case)
        run_trendextraction(writeoutput, mode, case)
        run_noderank(writeoutput, mode, case)
        run_plotting(writeoutput, mode, case)
        logging.info("Done with case: " + case)