"""
Tests basic weight calculation functions.

@author: Simon Streicher

"""

import multiprocessing
import unittest

from ranking.gaincalc import weightcalc


class TestWeightcalc(unittest.TestCase):

    def setUp(self):
        self.mode = 'tests'
        self.case = 'weightcalc_tests'

    def test_weightcalc_singleprocess(self):
        weightcalc(self.mode, self.case, False, True, True, False)

#    def test_weightcalc_multiprocess(self):
#        weightcalc(self.mode, self.case, False, False, False, True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    unittest.main()
