"""
@author: Simon Streicher

"""
import logging
import timeit
import json
import os

import config_setup
from ranking.gaincalc import weightcalc

import multiprocessing


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

if __name__ == '__main__':
    multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO)
    dataloc, _ = config_setup.get_locations()
    weightcalc_config = json.load(open(os.path.join(dataloc, 'config'
                                                    '_weightcalc' + '.json')))

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
        # TODO: For accurrate timing do it for the actual calculation only
        wrapped = wrapper(weightcalc, mode, case, writeoutput,
                          single_entropies, fftcalc, do_multiprocessing)
    print timeit.timeit(wrapped, number=1)
