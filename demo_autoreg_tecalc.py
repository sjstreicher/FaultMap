# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 01:27:15 2014

@author: Simon
"""

from autoreg import autoreg_gen
from transentropy import setup_infodynamics_te
from transentropy import vectorselection
from transentropy import calc_infodynamics_te
from transentropy import calc_custom_eq4_te
from transentropy import calc_custom_eq8_te
from transentropy import calc_custom_shu_te

import jpype
from sklearn import preprocessing

# TODO: This script must become a test case


def autoreg_datagen(delay, timelag, samples, sub_samples, k=1, l=1):
    """Generates autoreg data for a specific timelag (equal to
    prediction horison) for a set of autoregressive data.

    sub_samples is the amount of samples in the dataset used to calculate the
    transfer entropy between two vectors (taken from the end of the dataset).
    sub_samples <= samples

    Currently only supports k = 1; l = 1

    You can search through a set of timelags in an attempt to identify the
    original delay.
    The transfer entropy should have a maximum value when timelag = delay
    used to generate the autoregressive dataset.

    """

    data = autoreg_gen(samples, delay)

    [x_pred, x_hist, y_hist] = vectorselection(data, timelag,
                                               sub_samples, k, l)

    return x_pred, x_hist, y_hist


# Change location of jar to match yours:
infodynamicsloc = "infodynamics.jar"

# Start the JVM
# (add the "-Xmx" option with say 1024M if you get crashes
# due to not enough memory space)
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea",
               "-Djava.class.path=" + infodynamicsloc)

delay = 5
samples = 2500
sub_samples = 2000
for timelag in range(0, 11):
    print "Results for timelag of: ", str(timelag)
    [x_pred, x_hist, y_hist] = autoreg_datagen(delay, timelag,
                                               samples, sub_samples)

    # Normalize data
    # Not explicitly required as this is done by infodyns package if
    # setProperty("NORMALISE", "true" is called), but good practice
    # for general example.
    x_pred_norm = preprocessing.scale(x_pred, axis=1)
    x_hist_norm = preprocessing.scale(x_hist, axis=1)
    y_hist_norm = preprocessing.scale(y_hist, axis=1)

    # Calculate transfer entropy according to infodynamics method:

    teCalc = setup_infodynamics_te()

    result = calc_infodynamics_te(teCalc, x_hist_norm, y_hist_norm)
    print("Infodynamics TE result: %.4f bits" % (result))

    # Calculate transfer entropy according to custom
    # Lizier Eq. 4 implementation:

    #result = calc_custom_eq4_te(x_pred_norm, x_hist_norm, y_hist_norm, 10000)
    #print("Custom Eq. 4 TE result: %.4f bits" % (result))

    # Calculate transfer entropy according to custom
    # Lizier Eq. 8 implementation:

    result = calc_custom_eq8_te(x_pred_norm, x_hist_norm, y_hist_norm)
    print("Custom Eq. 8 TE result: %.4f bits" % (result))

    # It is observed that the calc_custom_eq4_te calculation approaches that
    # of calc_custom_eq8_te if the number of samples is increased.
    # This is expected from the transformation between between Lizier2008 Eq. 4
    # and Lizier2008 Eq. 8 and the fact that Mc integration instead of full
    # summation is used in the implementation of the Eq. 4 method.

    # Calculate transfer entropy according to custom Shu Eq. 2 implementation:
    # (This is very slow)
    # Do not believe it will be that helpful

    #result = calc_custom_shu_te(x_pred_norm, x_hist_norm, y_hist_norm)
    #print("Custom Shu TE result: %.4f bits" % (result))

jpype.shutdownJVM()
