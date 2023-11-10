# -*- coding: utf-8 -*-
"""Performs weight calculation for all cases specified in the
configuration file.

"""
# Standard modules
import json
import logging
import multiprocessing
import os

from faultmap import config_setup
from faultmap.gaincalc import weight_calc

if __name__ == "__main__":
    multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO)
    dataloc, configloc, _, _ = config_setup.get_locations()
    weightcalc_config = json.load(
        open(os.path.join(configloc, "config_weightcalc.json"))
    )

    # Flag indicating whether calculated results should be written to disk
    writeoutput = weightcalc_config["writeoutput"]
    # Flag indicating whether single signal entropy values for each
    # signal involved should be calculated
    single_entropies = weightcalc_config["calc_single_entropies"]
    # Provide the mode and case names to calculate
    mode = weightcalc_config["mode"]
    cases = weightcalc_config["cases"]
    fftcalc = weightcalc_config["fft_calc"]
    do_multiprocessing = weightcalc_config["multiprocessing"]

    for case in cases:
        weight_calc(
            mode,
            case,
            writeoutput,
            single_entropies,
            fftcalc,
            do_multiprocessing,
        )
