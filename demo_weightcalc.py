"""
@author: Simon Streicher

"""
import logging
import timeit
import json
import os

import config_setup
from ranking.gaincalc import weightcalc

logging.basicConfig(level=logging.INFO)

dataloc, _ = config_setup.get_locations()
weightcalc_config = json.load(open(os.path.join(dataloc, 'config'
                                                '_weightcalc' + '.json')))

writeoutput = weightcalc_config['writeoutput']
sigtest = weightcalc_config['sigtest']
mode = weightcalc_config['mode']
cases = weightcalc_config['cases']


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

for case in cases:
    wrapped = wrapper(weightcalc, mode, case, sigtest, writeoutput)
    print timeit.timeit(wrapped, number=1)
