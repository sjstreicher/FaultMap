# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 01:27:15 2014

@author: Simon
"""

from autoreg import autoreg_gen
from transentropy import vectorselection
from jpype import *
from sklearn import preprocessing

samples = 3000
delay = 5
data = autoreg_gen(samples, delay)

timelag = 5
sub_samples = 2000
[x_pred, x_hist, y_hist] = vectorselection(data, timelag, sub_samples)

# Normalize data
# Not explicitly required as this is done by infodyns package if
# setProperty("NORMALISE", "true" is called), but good practice
# for general example.
x_pred_norm = preprocessing.scale(x_pred, axis=1)
x_hist_norm = preprocessing.scale(x_hist, axis=1)
y_hist_norm = preprocessing.scale(y_hist, axis=1)

# Change location of jar to match yours:
jarLocation = "infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

sourceArray = y_hist_norm.tolist()[0]
destArray = x_hist_norm.tolist()[0]

# Create a TE calculator and run it:
teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
teCalc = teCalcClass()
# Normalise the individual variables
teCalc.setProperty("NORMALISE", "true")
 # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
teCalc.initialise(1, 0.5)
teCalc.setObservations(JArray(JDouble, 1)(sourceArray),
                       JArray(JDouble, 1)(destArray))

result = teCalc.computeAverageLocalOfObservations()
print("TE result %.4f bits" % (result))

shutdownJVM()
