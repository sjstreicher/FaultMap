"""Verifies the working of the transfer entropy calculation code by means of
an example on autoregressive data with known time delay.

Created on Mon Feb 24 14:56:25 2014

@author: Simon Streicher
"""

from transentropy import setup_infodynamics_te as te_info_setup
from transentropy import calc_infodynamics_te as te_info
from transentropy import calc_custom_shu_te as te_shu
from transentropy import calc_custom_eq8_te as te_eq8
from datagen import autoreg_datagen
from sklearn import preprocessing
import unittest
import jpype


class TestAutoregressiveTransferEntropy(unittest.TestCase):

    def setUp(self):
        """Generate list of entropies to test"""

        # Define number of samples to generate
        self.samples = 2500
        # Define number of samples to analyse
        self.sub_samples = 200
        # Delay in actul data
        self.delay = 5

        # Calculate transfer entropies in range of +/- 5 from actual delay
        # Use infodynamics package
        # Start JVM
        infodynamicsloc = "infodynamics.jar"
        if not jpype.isJVMStarted():
            # Start the JVM
            # (add the "-Xmx" option with say 1024M if you get crashes
            # due to not enough memory space)
            jpype.startJVM(jpype.getDefaultJVMPath(), "-ea",
                           "-Djava.class.path=" + infodynamicsloc)

        self.entropies_infodyn = []
        self.entropies_shu = []
        self.entropies_eq8 = []
        for timelag in range(self.delay-5, self.delay+6):
            print "Results for timelag of: ", str(timelag)
            [x_pred, x_hist, y_hist] = autoreg_datagen(self.delay, timelag,
                                                       self.samples,
                                                       self.sub_samples)

            # Normalize data
            # Not explicitly required as this is done by infodyns package if
            # setProperty("NORMALISE", "true" is called), but good practice
            # for general example.

            x_pred_norm = preprocessing.scale(x_pred, axis=1)
            x_hist_norm = preprocessing.scale(x_hist, axis=1)
            y_hist_norm = preprocessing.scale(y_hist, axis=1)

            # Calculate transfer entropy according to infodynamics method:

            # Get teCalc object
            teCalc = te_info_setup()

            result_infodyn = te_info(teCalc, x_hist_norm[0], y_hist_norm[0])
            self.entropies_infodyn.append(result_infodyn)
            print("Infodynamics TE result: %.4f bits" % (result_infodyn))

            result_eq8 = te_eq8(x_pred_norm, x_hist_norm, y_hist_norm)
            self.entropies_eq8.append(result_eq8)
            print("Eq8 TE result: %.4f bits" % (result_eq8))

            result_shu = te_shu(x_pred_norm, x_hist_norm, y_hist_norm)
            self.entropies_shu.append(result_shu)
            print("Shu TE result: %.4f bits" % (result_shu))

        print self.entropies_infodyn
        print self.entropies_eq8
        print self.entropies_shu

    def test_peakentropy_infodyn(self):
        maxval = max(self.entropies_infodyn)
        # The maximum lags with one sample
        delayedval = self.entropies_infodyn[self.delay-1]
        self.assertEqual(maxval, delayedval)

    def test_peakentropy_shu(self):
        maxval = max(self.entropies_shu)
        # The maximum lags with one sample
        delayedval = self.entropies_infodyn[self.delay]
        self.assertEqual(maxval, delayedval)

if __name__ == '__main__':
    unittest.main()
