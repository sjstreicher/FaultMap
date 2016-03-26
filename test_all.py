"""
@author: Simon Streicher

"""

import unittest

from ranking.gaincalc import weightcalc
from ranking.data_processing import result_reconstruction
import multiprocessing


class TestWeightcalc(unittest.TestCase):

    def setUp(self):
        self.mode = 'tests'
        self.case = 'weightcalc_tests'
        self.writeoutput = True

    def test_weightcalc_singleprocess(self):
        weightcalc(self.mode, self.case, self.writeoutput, False, False, False)

#    def test_weightcalc_multiprocess(self):
#        weightcalc(self.mode, self.case, False, False, False, True)


class TestCreateArrays(unittest.TestCase):

    def setUp(self):
        self.mode = 'tests'
        self.case = 'weightcalc_tests'
        self.writeoutput = True

    def test_createarrays_singleprocess(self):
        result_reconstruction(self.mode, self.case, self.writeoutput)

#    def test_weightcalc_multiprocess(self):
#        weightcalc(self.mode, self.case, False, False, False, True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    unittest.main()
