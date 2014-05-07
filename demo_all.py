# -*- coding: utf-8 -*-
"""This script runs all demos and is used for coverage analysis

Created on Sun Mar 16 17:29:06 2014

@author: Simon
"""

import glob
import logging

logging.basicConfig(level=logging.INFO)


def runall(pattern, exclude=[]):
    for f in glob.glob(pattern):
        if f not in exclude:
            logging.info('Running {}'.format(f))
            execfile(f)

# Demos
runall('demo_*.py', ['demo_all.py'])
# Tests
runall('test_*.py')
