"""Tests all functionality. Aims for full coverage.

"""

import multiprocessing

from run_full import run_all


def test_full_analysis():
    """Performs and end-to-end run on the test cases."""
    run_all("test")


# class TestWeightCalculation(unittest.TestCase):
#
#    def setUp(self):
#        self.mode = 'test'
#        self.case = 'fulldemo'
#
#    def test_weightcalc_singleprocess(self):
#        weightcalc(self.mode, self.case, True, True, True, False)
#
#    def test_weightcalc_multiprocess(self):
#        weightcalc(self.mode, self.case, False, False, False, True)
#
#
# class TestCreateArrays(unittest.TestCase):
#
#    def setUp(self):
#        self.mode = 'test'
#        self.case = 'fulldemo'
#        weightcalc(self.mode, self.case, True, False, False, True)
#
#    def test_createarrays(self):
#        result_reconstruction(self.mode, self.case)
#
#
# class TestTrendExtraction(unittest.TestCase):
#
#    def setUp(self):
#        self.mode = 'test'
#        self.case = 'fulldemo'
#        weightcalc(self.mode, self.case, True, False, False, True)
#        result_reconstruction(self.mode, self.case)
#
#    def test_trendextraction(self):
#        trend_extraction(self.mode, self.case, True)
#
#
# class TestNodeRanking(unittest.TestCase):
#
#    def setUp(self):
#        self.mode = 'test'
#        self.case = 'fulldemo'
#        weightcalc(self.mode, self.case, True, False, False, True)
#        result_reconstruction(self.mode, self.case)
#        trend_extraction(self.mode, self.case)
#
#    def test_noderanking(self):
#        noderankcalc(self.mode, self.case, True)
#
#
# class TestGrapReduce(unittest.TestCase):
#
#    def setUp(self):
#        self.mode = 'test'
#        self.case = 'fulldemo'
#        weightcalc(self.mode, self.case, True, False, False, True)
#        result_reconstruction(self.mode, self.case)
#        trend_extraction(self.mode, self.case, True)
#        noderankcalc(self.mode, self.case, True)
#
#    def test_graphreduce(self):
#        reducegraph(self.mode, self.case, True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    test_full_analysis()
