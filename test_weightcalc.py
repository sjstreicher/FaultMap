"""
@author: Simon Streicher

"""

import unittest

from ranking.gaincalc import weightcalc
import multiprocessing


class TestWeightcalc(unittest.TestCase):

    def setUp(self):
        self.mode = 'tests'
        self.case = 'weightcalc_tests'

    def test_weightcalc(self):
        weightcalc(self.mode, self.case)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    unittest.main()
