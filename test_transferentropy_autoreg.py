"""Verifies the working of the transfer entropy calculation code by means of
an example on autoregressive data with known time delay.

Created on Mon Feb 24 14:56:25 2014

@author: Simon Streicher
"""

from transentropy import calc_infodynamics_te as te_info
from transentropy import calculate_te as te
import unittest
import numpy as np


class TestAutoregressiveTransferEntropy(unittest.TestCase):
    def setUp(self):
        """Generate list of entropies to test"""
        # Randomly select delay in actual data
        self.delay = np.random.randint(10, 15)
        # Calculate transfer entropies in range of +/- 5 from actual delay
#        self.entropies = np.zeros(11)
#        for index, timelag in enumerate(range(self.delay-5, self.delay+6)):
#            self.entropies[index] = te(self.delay, timelag, 3000, 2000, 10000)
#        self.entropies_infodyn = [te_info(self.delay, timelag, 3000, 2000) for
#                          timelag in range(self.delay-5, self.delay+6)]
#        print self.entropies_infodyn
        self.entropies = [te(self.delay, timelag, 3000, 2000, 1000)[0] for
                          timelag in range(self.delay-5, self.delay+6)]
        print self.entropies

    def test_peakentropy(self):
        maxval = max(self.entropies)
        delayedval = self.entropies[5]
        self.assertEqual(maxval, delayedval)

#    def test_valueinrange(self):
#        for entropy in self.entropies:
#            self.assertLessEqual(entropy, 1)
#            self.assertGreaterEqual(entropy, 0)

if __name__ == '__main__':
    unittest.main()
