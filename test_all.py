"""
@author: Simon Streicher

"""

import multiprocessing
import unittest

from ranking.data_processing import result_reconstruction
from ranking.gaincalc import weightcalc


class TestWeightcalc(unittest.TestCase):

    def setUp(self):
        self.mode = 'tests'
        self.case = 'weightcalc_tests'
        self.writeoutput = True

    def test_weightcalc_singleprocess(self):
        weightcalc(self.mode, self.case, self.writeoutput, False, False, False)

    def test_weightcalc_multiprocess(self):
        weightcalc(self.mode, self.case, False, False, False, True)


class TestCreateArrays(unittest.TestCase):

    def setUp(self):
        self.mode = 'tests'
        self.case = 'weightcalc_tests'
        self.writeoutput = True
        weightcalc(self.mode, self.case, self.writeoutput, False, False, False)

    def test_createarrays_singleprocess(self):
        result_reconstruction(self.mode, self.case, self.writeoutput)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    unittest.main()
