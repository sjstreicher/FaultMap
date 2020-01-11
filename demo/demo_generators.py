# -*- coding: utf-8 -*-
"""Generates and plots test data.

"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from test import datagen

sns.set_style("darkgrid")

# Test sinusoid_shift_gen
sample_length = 1000
testdata = pd.DataFrame(datagen.sinusoid_shift_gen([sample_length]))

# Plotting variables

y_label = "Value"
x_label = "Sample"

plt.figure(1, (12, 6))

for dataindex in testdata:
    plt.plot(
        testdata.index.values,
        testdata[dataindex],
        "-",
        label=r"{}".format(testdata.columns.values[dataindex]),
    )

plt.ylabel(y_label, fontsize=14)
plt.xlabel(x_label, fontsize=14)
plt.legend()

# if graphdata.axis_limits is not False:
#    plt.axis(graphdata.axis_limits)
#
# plt.savefig(os.path.join(savedir, '{}_fft.pdf'.format(scenario)))
# plt.close()

plt.show()
