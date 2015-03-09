# -*- coding: utf-8 -*-
"""This script runs all tests and demo's and is used for coverage analysis
"""

import glob
import logging

logging.basicConfig(level=logging.INFO)


def runall(pattern, exclude=[]):
    for f in glob.glob(pattern):
        if f not in exclude:
            logging.info('Running {}'.format(f))
            execfile(f)

# Tests
runall('test_*.py', ['test_all.py'])
# Demos
runall('demo_*.py')
