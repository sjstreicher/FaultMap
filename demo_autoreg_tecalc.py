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

import jpype
from sklearn import preprocessing

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
sub_samples = 1000

infodynamics_results = np.zeros(len(range(0, 11)))

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

    teCalc = setup_infodynamics_te(True, 'kraskov')

    result = calc_infodynamics_te(teCalc, x_hist_norm[0], y_hist_norm[0])
    print("Infodynamics TE result: %.4f bits" % (result))

    infodynamics_results[timelag] = result

# Plot results over time delay
#fig, ax = plt.subplots()
#ax.plot(range(0, 11), infodynamics_results)
