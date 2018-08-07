# -*- coding: utf-8 -*-
"""Performs all analysis steps for all cases specified in the
configuration file.

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
from ranking.graphreduce import reducegraph
from plotting.plotter import plotdraw


# TODO: Move to class object
# TODO: Perform analysis on scenario level inside class object


def run_weightcalc(configloc, writeoutput, mode, case, robust):
    with open(os.path.join(configloc, "config_weightcalc" + ".json")) as f:
        weightcalc_config = json.load(f)
    f.close()

    # Flag indicating whether single signal entropy values for each
    # signal involved should be calculated
    single_entropies = weightcalc_config["calc_single_entropies"]
    # Flag indicating whether
    fftcalc = weightcalc_config["fft_calc"]
    do_multiprocessing = weightcalc_config["multiprocessing"]

    if robust:
        try:
            weightcalc(
                mode, case, writeoutput, single_entropies, fftcalc, do_multiprocessing
            )
        except:
            raise RuntimeError("Weight calculation failed for case: " + case)
    else:
        weightcalc(
            mode, case, writeoutput, single_entropies, fftcalc, do_multiprocessing
        )

    return None


def run_createarrays(writeoutput, mode, case, robust):

    if robust:
        try:
            # Needs to execute twice for nosigtest cases if derived from
            # sigtest cases
            # TODO: Remove this requirement
            result_reconstruction(mode, case, writeoutput)
            result_reconstruction(mode, case, writeoutput)
        except:
            raise RuntimeError("Array creation failed for case: " + case)
    else:
        result_reconstruction(mode, case, writeoutput)
        result_reconstruction(mode, case, writeoutput)

    return None


def run_trendextraction(writeoutput, mode, case, robust):

    if robust:
        try:
            trend_extraction(mode, case, writeoutput)
        except:
            raise RuntimeError("Trend extraction failed for case: " + case)
    else:
        trend_extraction(mode, case, writeoutput)

    return None


def run_noderank(writeoutput, mode, case, robust):

    if robust:
        try:
            noderankcalc(mode, case, writeoutput)
        except:
            raise RuntimeError("Node ranking failed for case: " + case)
    else:
        noderankcalc(mode, case, writeoutput)

    return None


def run_graphreduce(writeoutput, mode, case, robust):

    if robust:
        try:
            reducegraph(mode, case, writeoutput)
        except:
            raise RuntimeError("Graph reduction failed for case: " + case)
    else:
        reducegraph(mode, case, writeoutput)


def run_plotting(writeoutput, mode, case, robust):

    if robust:
        try:
            plotdraw(mode, case, writeoutput)
        except:
            raise RuntimeError("Plotting failed for case: " + case)
    else:
        plotdraw(mode, case, writeoutput)

    return None


def run_all(mode, robust=False):
    _, configloc, _, _ = config_setup.get_locations(mode)
    with open(os.path.join(configloc, "config_full" + ".json")) as f:
        fullrun_config = json.load(f)
    f.close()

    # Flag indicating whether calculated results should be written to disk
    writeoutput = fullrun_config["writeoutput"]
    # Provide the mode and case names to calculate
    mode = fullrun_config["mode"]
    cases = fullrun_config["cases"]

    for case in cases:
        logging.info("Now attempting case: " + case)
        run_weightcalc(configloc, writeoutput, mode, case, robust)
        run_createarrays(writeoutput, mode, case, robust)
        run_trendextraction(writeoutput, mode, case, robust)
        run_noderank(writeoutput, mode, case, robust)
        run_graphreduce(writeoutput, mode, case, robust)
        run_plotting(writeoutput, mode, case, robust)
        logging.info("Done with case: " + case)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO)

    run_all("cases")
