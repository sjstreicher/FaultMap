# -*- coding: utf-8 -*-
"""Demonstrates single signal entropy calculations.

"""

from test.datagen import autoreg_gen

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from faultmap import transentropy
from faultmap.data_processing import split_tsdata

sns.set_style("darkgrid")

INFODYNAMICS_LOCATION = "infodynamics.jar"
ESTIMATOR = "kernel"
# ESTIMATOR = 'gaussian'

SAMPLES = 1e6

ALPHA = 0.9
NOISE_RATIO = 0.1
DELAY = 10

BOX_SIZE = 100
BOX_NUM = 10

SAMPLING_RATE = 1
KERNEL_BANDWIDTH = 0.5

sampledata = autoreg_gen([SAMPLES, DELAY, ALPHA, NOISE_RATIO])

boxes = split_tsdata(sampledata, SAMPLING_RATE, BOX_SIZE, BOX_NUM)

CAUSE_VAR_INDEX = 0
AFFECTED_VAR_INDEX = 1


def gaussian_entropy(data):
    """Returns entropy of Gaussian signal in bits

    Args:
        data:

    Returns:

    """
    variance = np.var(data)
    entropy = 0.5 * np.log2(2.0 * np.pi * np.e * variance)
    return entropy


def silverman_bandwidth(var_data):
    return 1.06 * np.std(var_data) * (len(var_data) ** (-1 / 5))


def calculate_entropies(var_data, estimator=ESTIMATOR):
    # kernel_bandwidth = silverman_bandwidth(var_data)
    kernel_bandwidth = 0.8
    print(f"Kernel bandwidth: {str(kernel_bandwidth)}")
    # Setup Java class for infodynamics toolkit
    entropy_calc = transentropy.setup_infodynamics_entropy(
        INFODYNAMICS_LOCATION, estimator, kernel_bandwidth
    )
    signal_entropy_kernel = transentropy.calc_infodynamics_entropy(
        entropy_calc, var_data, estimator
    )
    signal_entropy_gaussian = gaussian_entropy(var_data)

    return signal_entropy_kernel, signal_entropy_gaussian


# corr_boxresults = []
# te_boxresults = []
signalent_cause_kernel_boxresults = []
signalent_cause_gaussian_boxresults = []
signalent_affected_kernel_boxresults = []
signalent_affected_gaussian_boxresults = []
for boxindex, box in enumerate(boxes):
    print("Now processing box: " + str(boxindex + 1))
    vardata_cause = box.T[0]
    vardata_effect = box.T[1]

    signalent_cause_kernel, signalent_cause_gaussian = calculate_entropies(
        vardata_cause
    )

    (
        signalent_affected_kernel,
        signalent_affected_gaussian,
    ) = calculate_entropies(vardata_effect)

    signalent_cause_kernel_boxresults.append(signalent_cause_kernel)
    signalent_cause_gaussian_boxresults.append(signalent_cause_gaussian)
    signalent_affected_kernel_boxresults.append(signalent_affected_kernel)
    signalent_affected_gaussian_boxresults.append(signalent_affected_gaussian)

fig, ax1 = plt.subplots()
ax1.plot(range(len(boxes)), signalent_cause_kernel_boxresults, label="cause kernel")
ax1.plot(
    range(len(boxes)),
    signalent_cause_gaussian_boxresults,
    label="cause gaussian",
)
ax1.set_xlabel("Box")
ax1.set_ylabel("Signal entropy")
plt.legend()
plt.show()

fig, ax1 = plt.subplots()
ax1.plot(
    range(len(boxes)),
    signalent_affected_kernel_boxresults,
    label="affected kernel",
)
ax1.plot(
    range(len(boxes)),
    signalent_affected_gaussian_boxresults,
    label="affected gaussian",
)
ax1.set_xlabel("Box")
ax1.set_ylabel("Signal entropy")
plt.legend()
plt.show()
