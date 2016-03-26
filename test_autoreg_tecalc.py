# -*- coding: utf-8 -*-
"""Verifies the working of the transfer entropy calculation code by means of
an example on autoregressive data with known time delay.

@author: Simon Streicher
"""

import unittest

import jpype
from sklearn import preprocessing

from transentropy import calc_infodynamics_te as te_info
from datagen import autoreg_datagen


class TestAutoregressiveTransferEntropy(unittest.TestCase):

    def setUp(self):
        """Generate list of entropies to test"""

        # Define number of samples to generate
        self.samples = 2500
        # Define number of samples to analyse
        self.sub_samples = 1000
        # Delay in actual data
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

    def test_peakentropy_infodyn_kernel(self):
        self.entropies_infodyn_kernel = []
        for timelag in range(self.delay-5, self.delay+6):
            print "Results for timelag of: ", str(timelag)
            [x_pred, x_hist, y_hist] = autoreg_datagen(self.delay, timelag,
                                                       self.samples,
                                                       self.sub_samples)

            # Normalize data
            # Not explicitly required as this is done by infodyns package if
            # setProperty("NORMALISE", "true" is called), but good practice
            # for general example.

            x_hist_norm = preprocessing.scale(x_hist, axis=1)
            y_hist_norm = preprocessing.scale(y_hist, axis=1)

            # Calculate transfer entropy according to infodynamics method:

            result_infodyn, _ = te_info('infodynamics.jar',
                                        True, 'kernel',
                                        x_hist_norm[0], y_hist_norm[0])
            self.entropies_infodyn_kernel.append(result_infodyn)
            print("Infodynamics TE result: %.4f bits" % result_infodyn)

        print self.entropies_infodyn_kernel

        maxval = max(self.entropies_infodyn_kernel)
        # The maximum lags with one sample
        delayedval = self.entropies_infodyn_kernel[self.delay-1]
        self.assertEqual(maxval, delayedval)

    def test_peakentropy_infodyn_kraskov_noautoembed(self):
        self.entropies_infodyn_kraskov = []
        for timelag in range(self.delay-5, self.delay+6):
            print "Results for timelag of: ", str(timelag)
            [x_pred, x_hist, y_hist] = autoreg_datagen(self.delay, timelag,
                                                       self.samples,
                                                       self.sub_samples)
            # Normalize data
            # Not explicitly required as this is done by infodyns package if
            # setProperty("NORMALISE", "true" is called), but good practice
            # for general example.

            x_hist_norm = preprocessing.scale(x_hist, axis=1)
            y_hist_norm = preprocessing.scale(y_hist, axis=1)

            # Calculate transfer entropy according to infodynamics method:

            result_infodyn, [significance, properties] = \
                te_info('infodynamics.jar', True, 'kraskov',
                        x_hist_norm[0], y_hist_norm[0],
                        test_significance=True,
                        auto_embed=False)
            self.entropies_infodyn_kraskov.append(result_infodyn)
            print("Infodynamics TE result: %.4f nats" % result_infodyn)

            print properties
            print significance

        print self.entropies_infodyn_kraskov

        maxval = max(self.entropies_infodyn_kraskov)
        # The maximum lags with one sample
        delayedval = self.entropies_infodyn_kraskov[self.delay-1]
        self.assertEqual(maxval, delayedval)

    def test_peakentropy_infodyn_kraskov_autoembed(self):
        self.entropies_infodyn_kraskov = []
        for timelag in range(self.delay-5, self.delay+6):
            print "Results for timelag of: ", str(timelag)
            [x_pred, x_hist, y_hist] = autoreg_datagen(self.delay, timelag,
                                                       self.samples,
                                                       self.sub_samples)
            # Normalize data
            # Not explicitly required as this is done by infodyns package if
            # setProperty("NORMALISE", "true" is called), but good practice
            # for general example.

            x_hist_norm = preprocessing.scale(x_hist, axis=1)
            y_hist_norm = preprocessing.scale(y_hist, axis=1)

            # Calculate transfer entropy according to infodynamics method:

            result_infodyn, [significance, properties] = \
                te_info('infodynamics.jar', True, 'kraskov',
                        x_hist_norm[0], y_hist_norm[0],
                        test_significance=True,
                        auto_embed=True)
            self.entropies_infodyn_kraskov.append(result_infodyn)
            print("Infodynamics TE result: %.4f nats" % result_infodyn)

            print properties
            print significance

        print self.entropies_infodyn_kraskov

        maxval = max(self.entropies_infodyn_kraskov)
        # The maximum lags with one sample
        delayedval = self.entropies_infodyn_kraskov[self.delay-1]
        self.assertEqual(maxval, delayedval)

if __name__ == '__main__':
    unittest.main()
