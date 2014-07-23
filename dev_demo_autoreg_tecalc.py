# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 01:27:15 2014

@author: Simon
"""

import numpy as np
import matplotlib.pyplot as plt

from datagen import autoreg_datagen
from transentropy import setup_infodynamics_te
from transentropy import calc_infodynamics_te
from dev_te_python import calc_sklearn_te
from dev_te_python import calc_scipy_te
from dev_te_python import calc_custom_shu_te

#from transentropy import setup_infodynamics_te as te_info_setup
#from transentropy import calc_infodynamics_te as te_info
#from transentropy import calc_custom_shu_te as te_shu
#from transentropy import calc_custom_eq8_te as te_eq8
from sklearn import preprocessing
#import unittest
import jpype

# Change location of jar to match yours:
infodynamicsloc = "infodynamics.jar"

if not jpype.isJVMStarted():
    # Start the JVM
    # (add the "-Xmx" option with say 1024M if you get crashes
    # due to not enough memory space)
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea",
                   "-Djava.class.path=" + infodynamicsloc)

delay = 5
samples = 2500
# Low value selected for demonstration purposes only
sub_samples = 100

infodynamics_results = np.zeros(len(range(0, 11)))
sklearn_results = np.zeros_like(infodynamics_results)
scipy_results = np.zeros_like(infodynamics_results)
shu_results = np.zeros_like(infodynamics_results)

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

    result = calc_infodynamics_te(teCalc, x_hist_norm[0], y_hist_norm[0])
    print("Infodynamics TE result: %.4f bits" % (result))

    infodynamics_results[timelag] = result

    # Calculate transfer entropy according to sklearn method:

    sklearn_result = calc_sklearn_te(x_pred_norm[0], x_hist_norm[0],
                                   y_hist_norm[0])

    sklearn_results[timelag] = sklearn_result

    print("Custom sklearn TE result: %.4f bits" % (sklearn_result))

    # Calculate transfer entropy according to scipy method:

    scipy_result = calc_scipy_te(x_pred_norm[0], x_hist_norm[0],
                                   y_hist_norm[0])

    scipy_results[timelag] = scipy_result

    print("Custom scipy TE result: %.4f bits" % (scipy_result))

    # Calculate transfer entropy according to Shu method:

    shu_result = calc_custom_shu_te(x_pred_norm[0], x_hist_norm[0],
                                    y_hist_norm[0])

    shu_results[timelag] = shu_result

    print("Custom Shu TE result: %.4f bits" % (shu_result))

# Plot results over time delay
fig, ax = plt.subplots()
ax.plot(range(0, 11), infodynamics_results, label='infodynamics')
ax.plot(range(0, 11), sklearn_results, label='sklearn')
ax.plot(range(0, 11), scipy_results, label='scipy')
ax.plot(range(0, 11), shu_results, label='shu')
plt.show()
