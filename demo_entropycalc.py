# -*- coding: utf-8 -*-
"""
Demonstrates single signal entropy calculations.

@author: Simon Streicher
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import jpype

infodynamicsloc = "infodynamics.jar"
normalise = True
estimator = 'kernel'


#class SpoofClass(object):
#    
#    def __init__(self, infodynamicsloc, normalise):
#        self.infodynamicsloc = infodynamicsloc
#        self.normalise = normalise
#        
#weightcalcdata = SpoofClass(infodynamics_loc, True) 


sns.set_style("darkgrid")

from datagen import autoreg_gen
from ranking.data_processing import split_tsdata

import transentropy

samples = 200000

alpha = 0.9
noise_ratio = 0.1
delay = 10

boxsize = 30000
boxnum = 10

sampling_rate = 1

sampledata = autoreg_gen([samples, delay, alpha, noise_ratio])

boxes = split_tsdata(sampledata, sampling_rate, boxsize, boxnum)

causevarindex = 0
affectedvarindex = 1

# Setup Java class for infodynamics toolkit
entropyCalc = \
    transentropy.setup_infodynamics_entropy(
        infodynamicsloc, normalise, estimator=estimator,
        kernel_bandwidth=1.)

def gaussian_entropy(data):
    
    variance = np.var(data)
    entropy = 0.5 * np.log2(2.0 * np.pi * np.e * variance)
    return entropy

#corr_boxresults = []
#te_boxresults = []
signalent_boxresults = []
for boxindex, box in enumerate(boxes):
    print "Now processing box: " + str(boxindex + 1)
    vardata = box.T[1]
    
    dataArray = vardata.tolist()
    dataArrayJava = jpype.JArray(jpype.JDouble, 1)(dataArray)
    
    entropyCalc.setObservations(dataArrayJava)
    
    signalent = entropyCalc.computeAverageLocalOfObservations()
    signalent = gaussian_entropy(dataArray)
    
    signalent = transentropy.calc_infodynamics_entropy(
        entropyCalc, vardata)
    signalent_boxresults.append(signalent)

fig, ax1 = plt.subplots()
ax1.plot(range(len(boxes)), signalent_boxresults)
ax1.set_xlabel('Box')
ax1.set_ylabel('Signal entropy')


plt.show()
