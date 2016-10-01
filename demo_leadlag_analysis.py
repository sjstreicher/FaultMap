# -*- coding: utf-8 -*-
"""
Performs a lead/lag analysis on the full matrix provided.

@author: Simon Streicher
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from transentropy import calc_infodynamics_te

infodynamics_loc = "infodynamics.jar"
estimator = 'kraskov'

add_parameters = {'test_significance': False,
                  'significance_permutations': 30,
                  'auto_embed': False}
#test_significance = False
#significance_permutations = 30
#auto_embed = False


def tecalc_wrapper(causevardata, affectedvardata):

    te_fwd = calc_infodynamics_te(infodynamics_loc, estimator,
                                  causevardata.T, affectedvardata.T,
                                  **add_parameters)

    te_bwd = calc_infodynamics_te(infodynamics_loc, estimator,
                                  affectedvardata.T, causevardata.T,
                                  **add_parameters)

    return [te_fwd, te_bwd]


# Delays to test in the forwards as well as backwards directions


from datagen import autoreg_gen
from ranking.data_processing import split_tsdata

sns.set_style("darkgrid")


samples = 200000
delay = 10
test_delays = 50
testsize = 1000
startindex = 100
alpha = 0.9
noise_ratio = 0.1

boxsize = 1200
boxnum = 10

sampling_rate = 1

sampledata = autoreg_gen([samples, delay, alpha, noise_ratio])

boxes = split_tsdata(sampledata, sampling_rate, boxsize, boxnum)

causevarindex = 0
affectedvarindex = 1

delays = range(-test_delays, test_delays + 1)

corr_boxresults = []
te_boxresults = []
for boxindex, box in enumerate(boxes):
    print "Now processing box: " + str(boxindex + 1)
    corrvals = []
    tevals = []
    for delay in delays:

        causevardata = \
            (box[:, causevarindex]
                [startindex:startindex+testsize])

        affectedvardata = \
            (box[:, affectedvarindex]
                [startindex+delay:startindex+testsize+delay])

        corrval = np.corrcoef(causevardata.T, affectedvardata.T)[1, 0]

        teval = tecalc_wrapper(causevardata, affectedvardata)

        te_fwd = teval[0][0]
        te_bwd = teval[1][0]

        corrvals.append(corrval)
        tevals.append(teval)

    corr_boxresults.append(corrvals)
    te_boxresults.append(tevals)


te_fwd_list = []
te_bwd_list = []
te_diff_list = []
te_diff_boxresults = []
te_fwd_boxresults = []
te_bwd_boxresults = []
for te_boxresult in te_boxresults[0:1]:
    for delayindex, delay in enumerate(delays):
        te_fwd = te_boxresult[delayindex][0][0]
        te_bwd = te_boxresult[delayindex][1][0]
        te_diff = te_fwd - te_bwd
        te_fwd_list.append(te_fwd)
        te_bwd_list.append(te_bwd)
        te_diff_list.append(te_diff)
    te_diff_boxresults.append(te_diff_list)
    te_fwd_boxresults.append(te_fwd_list)
    te_bwd_boxresults.append(te_bwd_list)


fig, ax1 = plt.subplots()
for corr_boxresult in corr_boxresults[0:1]:
    ax1.plot(delays, corr_boxresult)
ax1.set_xlabel('Delay')
ax1.set_ylabel('CC')

ax2 = ax1.twinx()

for te_boxresult_index in range(len(te_diff_boxresults[0:1])):
    ax2.plot(delays, te_diff_boxresults[te_boxresult_index], 'g')
    ax2.plot(delays, te_fwd_boxresults[te_boxresult_index], 'b')
    ax2.plot(delays, te_bwd_boxresults[te_boxresult_index], 'r')
ax2.set_ylabel('TE diff')
plt.show()
