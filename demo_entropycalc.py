# -*- coding: utf-8 -*-
"""
Demonstrates single signal entropy calculations.

@author: Simon Streicher
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

infodynamicsloc = "infodynamics.jar"
estimator = 'kernel'
#estimator = 'gaussian'


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

samples = 1e6

alpha = 0.9
noise_ratio = 0.1
delay = 10

boxsize = 100
boxnum = 10

sampling_rate = 1
#kernel_bandwidth = 0.5

sampledata = autoreg_gen([samples, delay, alpha, noise_ratio])

boxes = split_tsdata(sampledata, sampling_rate, boxsize, boxnum)

causevarindex = 0
affectedvarindex = 1

def gaussian_entropy(data):
    "Returns entropy of Gaussian signal in bits"
    variance = np.var(data)
    entropy = 0.5 * np.log2(2.0 * np.pi * np.e * variance)
    return entropy

def silverman_bandwidth(vardata):
    return (1.06 * np.std(vardata) * (len(vardata) ** (-1/5)))

def calculate_entropies(vardata, estimator=estimator):
#    kernel_bandwidth = silverman_bandwidth(vardata)
    kernel_bandwidth = 0.8
    print("Kernel bandwidth: " + str(kernel_bandwidth))
    # Setup Java class for infodynamics toolkit
    entropyCalc = \
        transentropy.setup_infodynamics_entropy(
            infodynamicsloc, estimator, kernel_bandwidth)
    signalent_kernel = transentropy.calc_infodynamics_entropy(
        entropyCalc, vardata, estimator)
    signalent_gaussian = gaussian_entropy(vardata)

    return signalent_kernel, signalent_gaussian

#corr_boxresults = []
#te_boxresults = []
signalent_cause_kernel_boxresults = []
signalent_cause_gaussian_boxresults = []
signalent_affected_kernel_boxresults = []
signalent_affected_gaussian_boxresults = []
for boxindex, box in enumerate(boxes):
    print "Now processing box: " + str(boxindex + 1)
    vardata_cause = box.T[0]
    vardata_effect = box.T[1]

    signalent_cause_kernel, signalent_cause_gaussian = \
        calculate_entropies(vardata_cause)

    signalent_affected_kernel, signalent_affected_gaussian = \
        calculate_entropies(vardata_effect)

    signalent_cause_kernel_boxresults.append(signalent_cause_kernel)
    signalent_cause_gaussian_boxresults.append(signalent_cause_gaussian)
    signalent_affected_kernel_boxresults.append(signalent_affected_kernel)
    signalent_affected_gaussian_boxresults.append(signalent_affected_gaussian)

fig, ax1 = plt.subplots()
ax1.plot(range(len(boxes)), signalent_cause_kernel_boxresults, label='cause kernel')
ax1.plot(range(len(boxes)), signalent_cause_gaussian_boxresults, label='cause gaussian')
ax1.set_xlabel('Box')
ax1.set_ylabel('Signal entropy')
plt.legend()
plt.show()

fig, ax1 = plt.subplots()
ax1.plot(range(len(boxes)), signalent_affected_kernel_boxresults, label='affected kernel')
ax1.plot(range(len(boxes)), signalent_affected_gaussian_boxresults, label='affected gaussian')
ax1.set_xlabel('Box')
ax1.set_ylabel('Signal entropy')
plt.legend()
plt.show()
