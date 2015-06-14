"""This module contains the methods used in the calculation of transfer
entropy.

Created on Mon Feb 24 15:18:33 2014

@author: Simon Streicher
"""
import numpy as np
import jpype


def vectorselection(data, timelag, sub_samples, k=1, l=1):
    """Generates sets of vectors from tags time series data
    for calculating transfer entropy.

    For notation references see Shu2013.

    Takes into account the time lag (number of samples between vectors of the
    same variable).

    In this application the prediction horizon (h) is set to equal
    to the time lag.

    The first vector in the data array should be the samples of the variable
    to be predicted (x) while the second vector should be sampled of the vector
    used to make the prediction (y).

    sub_samples is the amount of samples in the dataset used to calculate the
    transfer entropy between two vectors and must satisfy
    sub_samples <= samples

    The required number of samples is extracted from the end of the vector.
    If the vector is longer than the number of samples specified plus the
    desired time lag then the remained of the data will be discarded.

    k refers to the dimension of the historical data to be predicted (x)

    l refers to the dimension of the historical data used
    to do the prediction (y)

    """
    _, sample_n = data.shape
    x_pred = data[0, sample_n-sub_samples:]
    x_pred = x_pred[np.newaxis, :]

    x_hist = np.zeros((k, sub_samples))
    y_hist = np.zeros((l, sub_samples))

    for n in range(1, k+1):
        # Original form according to Bauer (2007)
        # TODO: Provide for comparison
        # Modified form according to Shu & Zhao (2013)
        startindex = (sample_n - sub_samples) - timelag*(n - 1) - 1
        endindex = sample_n - timelag*(n - 1) - 1
        x_hist[n-1, :] = data[1, startindex:endindex]
    for m in range(1, l+1):
        startindex = (sample_n - sub_samples) - timelag*m - 1
        endindex = sample_n - timelag*m - 1
        y_hist[m-1:, :] = data[0, startindex:endindex]

    return x_pred, x_hist, y_hist


def setup_infodynamics_te(infodynamicsloc,
                          normalize, calcmethod, histlength=1):
    """Prepares the teCalc class of the Java Infodyamics Toolkit (JIDK)
    in order to calculate transfer entropy according to the kernel or Kraskov
    estimator method.

    Currently implemented for the case of k = 1 and l = 1 only.

    The embedding dimension of the destination or target variable (k) can
    easily be set by adjusting the histlength parameter.

    # TODO: Allow for different destination embedding dimensions of the source
    variable (l) by making use of the multivariable transfer entropy
    implementation - currently only available for the implementation making use
    of Kraskov MI estimators

    """

    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(),
                       "-Xms32M",
                       "-Xmx512M",
                       "-ea",
                       "-Djava.class.path=" + infodynamicsloc)

    if calcmethod == 'kernel':
        teCalcClass = \
            jpype.JPackage("infodynamics.measures.continuous.kernel") \
            .TransferEntropyCalculatorKernel
        teCalc = teCalcClass()
        # Set history length (Schreiber k=1)
        # Set kernel width to 0.5 normalised units
        teCalc.initialise(histlength, 0.5)

    elif calcmethod == 'kraskov':
        teCalcClass = \
            jpype.JPackage("infodynamics.measures.continuous.kraskov") \
            .TransferEntropyCalculatorKraskov
        teCalc = teCalcClass()
        # Set history length (Schreiber k=1)
        teCalc.initialise(histlength)
        # Use Kraskov parameter K=4 for 4 nearest points
        teCalc.setProperty("k", "4")

    # Normalise the individual variables if required
    if normalize:
        teCalc.setProperty("NORMALISE", "true")
    else:
        teCalc.setProperty("NORMALISE", "false")

    return teCalc


def calc_infodynamics_te(infodynamicsloc, normalize, calcmethod,
                         affected_data, causal_data):
    """Calculates the transfer entropy for a specific timelag (equal to
    prediction horison) between two sets of time series data.

    This implementation makes use of the infodynamics toolkit:
    https://code.google.com/p/information-dynamics-toolkit/

    The transfer entropy should have a maximum value when timelag = delay
    used to generate an autoregressive dataset, or will otherwise indicate the
    dead time between data indicating a causal relationship.

    """

    teCalc = setup_infodynamics_te(infodynamicsloc, normalize, calcmethod)

    sourceArray = causal_data.tolist()
    destArray = affected_data.tolist()

    sourceArrayJava = jpype.JArray(jpype.JDouble, 1)(sourceArray)
    destArrayJava = jpype.JArray(jpype.JDouble, 1)(destArray)

    teCalc.setObservations(sourceArrayJava,
                           destArrayJava)

    transentropy = teCalc.computeAverageLocalOfObservations()

    return transentropy


def setup_infodynamics_entropy(normalize):
    """Prepares the entropyCalc class of the Java Infodyamics Toolkit (JIDK)
    in order to calculate differential entropy (continuous signals) according
    to the box kernel estimation method.

    """

    entropyCalcClass = \
        jpype.JPackage("infodynamics.measures.continuous.kernel") \
        .EntropyCalculatorKernel
    entropyCalc = entropyCalcClass()
    # Set kernel width to 0.5 normalised units
    entropyCalc.initialise(0.5)

    # Normalise the individual variables if required
    if normalize:
        entropyCalc.setProperty("NORMALISE", "true")
    else:
        entropyCalc.setProperty("NORMALISE", "false")

    entropyCalcClass = None
    del entropyCalcClass
    jpype.java.lang.System.gc()

    return entropyCalc


def calc_infodynamics_entropy(entropyCalc, data):
    """Estimates the entropy of a single signal.

    Makes use of the box kernel estimation method.
    """

    dataArray = data.tolist()
    dataArrayJava = jpype.JArray(jpype.JDouble, 1)(dataArray)

    entropyCalc.setObservations(dataArrayJava)

    entropy = entropyCalc.computeAverageLocalOfObservations()

    return entropy
