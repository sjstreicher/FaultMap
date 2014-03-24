"""Verifies the working of the transfer entropy calculation code by means of
an example on autoregressive data with known time delay.

Created on Mon Feb 24 14:56:25 2014

@author: Simon Streicher
"""

from transentropy import calc_infodynamics_te as te_info
from transentropy import calc_custom_te as te_custom
import unittest


class TestAutoregressiveTransferEntropy(unittest.TestCase):
    def setUp(self):
        """Generate list of entropies to test"""
        # Randomly select delay in actual data
        self.delay = 13
        # Calculate transfer entropies in range of +/- 5 from actual delay
        # Use infodynamics package
        self.entropies_infodyn = [te_info(self.delay, timelag, 3000, 2000) for
                                  timelag in range(self.delay-5, self.delay+6)]
        print self.entropies_infodyn
        self.entropies_custom = [te_custom(self.delay, timelag,
                                           3000, 2000, 1000)[0] for
                                 timelag in range(self.delay-5, self.delay+6)]
        print self.entropies_custom

    def test_peakentropy_infodyn(self):
        maxval = max(self.entropies_infodyn)
        delayedval = self.entropies_infodyn[5]
        self.assertEqual(maxval, delayedval)

    def test_peakentropy_custom(self):
        maxval = max(self.entropies_custom)
        delayedval = self.entropies_custom[5]
        self.assertEqual(maxval, delayedval)

if __name__ == '__main__':
    unittest.main()
