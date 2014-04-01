# -*- coding: utf-8 -*-
"""This script runs all demos and is used for coverage analysis

Created on Sun Mar 16 17:29:06 2014

@author: Simon
"""

# Demos
execfile("demo_looprank.py")
execfile("demo_weightcalc.py")
execfile("demo_autoreg_tecalc.py")

# Tests
execfile("test_looprank.py")
execfile("test_weightcalc.py")
execfile("test_tecalc.py")
