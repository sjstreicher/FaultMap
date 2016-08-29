"""
Tests basic node ranking functions.

@author: Simon Streicher

"""

import multiprocessing
import unittest

from ranking.noderank import noderankcalc


class TestNoderank(unittest.TestCase):

    def setUp(self):
        self.mode = 'tests'
        self.case = 'noderank_tests'

    def test_noderank(self):
        noderankcalc(self.mode, self.case, False, False)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    unittest.main()
