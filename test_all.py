# -*- coding: utf-8 -*-
"""Tests all functionality. Aims for full coverage.

"""

import multiprocessing
import unittest

from ranking.gaincalc import weightcalc
from ranking.data_processing import result_reconstruction
from ranking.data_processing import trend_extraction
from ranking.noderank import noderankcalc
from ranking.graphreduce import reducegraph
from plotting.plotter import plotdraw


class TestFullDemo(unittest.TestCase):

    def setUp(self):
        self.mode = 'tests'
        self.case = 'quickdemo'

    def test_full_analysis(self):
        weightcalc(self.mode, self.case, True, False, False, False)
        result_reconstruction(self.mode, self.case, True)
        trend_extraction(self.mode, self.case, True)
        noderankcalc(self.mode, self.case, True)
        reducegraph(self.mode, self.case, True)
        plotdraw(self.mode, self.case, True)


#class TestWeightCalculation(unittest.TestCase):
#
#    def setUp(self):
#        self.mode = 'tests'
#        self.case = 'fulldemo'
#
#    def test_weightcalc_singleprocess(self):
#        weightcalc(self.mode, self.case, True, True, True, False)
#
#    def test_weightcalc_multiprocess(self):
#        weightcalc(self.mode, self.case, False, False, False, True)
#
#
#class TestCreateArrays(unittest.TestCase):
#
#    def setUp(self):
#        self.mode = 'tests'
#        self.case = 'fulldemo'
#        weightcalc(self.mode, self.case, True, False, False, True)
#
#    def test_createarrays(self):
#        result_reconstruction(self.mode, self.case, True)
#
#
#class TestTrendExtraction(unittest.TestCase):
#
#    def setUp(self):
#        self.mode = 'tests'
#        self.case = 'fulldemo'
#        weightcalc(self.mode, self.case, True, False, False, True)
#        result_reconstruction(self.mode, self.case, True)
#
#    def test_trendextraction(self):
#        trend_extraction(self.mode, self.case, True)
#
#
#class TestNodeRanking(unittest.TestCase):
#
#    def setUp(self):
#        self.mode = 'tests'
#        self.case = 'fulldemo'
#        weightcalc(self.mode, self.case, True, False, False, True)
#        result_reconstruction(self.mode, self.case, True)
#        trend_extraction(self.mode, self.case, True)
#
#    def test_noderanking(self):
#        noderankcalc(self.mode, self.case, True)
#
#
#class TestGrapReduce(unittest.TestCase):
#
#    def setUp(self):
#        self.mode = 'tests'
#        self.case = 'fulldemo'
#        weightcalc(self.mode, self.case, True, False, False, True)
#        result_reconstruction(self.mode, self.case, True)
#        trend_extraction(self.mode, self.case, True)
#        noderankcalc(self.mode, self.case, True)
#
#    def test_graphreduce(self):
#        reducegraph(self.mode, self.case, True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    unittest.main()
