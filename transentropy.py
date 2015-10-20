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
                          normalize, calcmethod, **parameters):
    """Prepares the teCalc class of the Java Infodyamics Toolkit (JIDT)
    in order to calculate transfer entropy according to the kernel or Kraskov
    estimator method. Also supports discrete transfer entropy calculation.

    The embedding dimension of the destination or target variable (k) can
    easily be set by adjusting the histlength parameter.

    # TODO: Allow for different destination embedding dimensions of the source
    variable (l) by making use of the multivariable transfer entropy
    implementation - currently only available for the implementation making use
    of Kraskov MI estimators

    # UPDATE: (Still needs to be verified) The continous Kraskov estimator
    with the Ragwitz criterion is the basis for all results unless mentioned
    otherwise as this is the theoretical best automated procedure that is
    available in JIDT 1.3.
    """

    # TODO: Allow for automated embedding dimension algorithms to be applied

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

        # Normalise the individual variables if required
        if normalize:
            teCalc.setProperty("NORMALISE", "true")
        else:
            teCalc.setProperty("NORMALISE", "false")

        # Parameter definitions - refer to JIDT Javadocs
        # k - destination embedded history length (Schreiber k=1)
        # kernelWidth - if NORMALISE_PROP_NAME property has been set,
        # then this kernel width corresponds to the number of
        # standard deviations from the mean (otherwise it is an absolute value)

        if ('k' in parameters):
            k = parameters['k']
        else:
            k = 1
#            print "k default of 1 is used"

        if ('kernel_width' in parameters):
            kernelWidth = parameters['kernel_width']
        else:
            kernelWidth = 0.5
#            print "kernelWidth default of 0.5 is used"

        teCalc.initialise(k, kernelWidth)

    elif calcmethod == 'kraskov':
        """The Kraskov method is the recommended method and also provides
        methods for auto-embedding. The Ragqitz criterion auto-embedding method
        will be enabled as the default.

        Methods for returning the k, k_tau, l and l_tau used will be
        implemented.

        """

        teCalcClass = \
            jpype.JPackage("infodynamics.measures.continuous.kraskov") \
            .TransferEntropyCalculatorKraskov
        teCalc = teCalcClass()
        # Parameter definitions - refer to JIDT javadocs

        # k - embedding length of destination past history to consider
        # k_tau - embedding delay for the destination variable
        # l - embedding length of source past history to consider
        # l_tau - embedding delay for the source variable
        # delay - time lag between last element of source and destination
        # next value

        # Normalise the individual variables if required
        if normalize:
            teCalc.setProperty("NORMALISE", "true")
        else:
            teCalc.setProperty("NORMALISE", "false")

        if ('auto_embed' in parameters):
            auto_embed = parameters['auto_embed']
            if auto_embed is True:
                # Enable the Ragwitz criterion
                # Enable source as well as destination embedding due to the
                # nature of our data.
                # Use a maximum history and tau search of 5
                teCalc.setProperty("AUTO_EMBED_METHOD", "RAGWITZ")
                teCalc.setProperty("AUTO_EMBED_K_SEARCH_MAX", "5")
                teCalc.setProperty("AUTO_EMBED_TAU_SEARCH_MAX", "5")

        # Note: If setting the delay is needed to be changed on each iteration,
        # it may be best to do this outside the loop and initialise teCalc
        # after each change.

        if ('delay' in parameters):
            delay = parameters['delay']
            teCalc.setProperty("DELAY", str(delay))

        # Allow for manual override

        if ('k_history' in parameters):
            k_history = parameters['k_history']
            teCalc.setProperty("k_HISTORY", str(k_history))

        if ('k_tau' in parameters):
            k_tau = parameters['k_tau']
            teCalc.setProperty("k_TAU", str(k_tau))

        if ('l_history' in parameters):
            l_history = parameters['l_history']
            teCalc.setProperty("l_HISTORY", str(l_history))

        if ('l_tau' in parameters):
            l_tau = parameters['l_tau']
            teCalc.setProperty("l_TAU", str(l_tau))

        teCalc.initialise()

    elif calcmethod == 'discrete':
        teCalcClass = \
            jpype.JPackage("infodynamics.measures.discrete") \
            .TransferEntropyCalculatorDiscrete
        # Parameter definitions - refer to JIDT javadocs
        # base - number of quantisation levels for each variable
        # binary variables are in base-2
        # destHistoryEmbedLength - embedded history length of the
        # destination to condition on - this is k in Schreiber's notation
        # sourceHistoryEmbeddingLength - embedded history length of the source
        # to include - this is l in Schreiber's notation
        # TODO: Allow these settings to be defined by configuration file

        if ('base' in parameters):
            base = parameters['base']
        else:
            base = 2
            print "base default of 2 (binary) is used"

        if ('destHistoryEmbedLength' in parameters):
            destHistoryEmbedLength = parameters['destHistoryEmbedLength']
        else:
            destHistoryEmbedLength = 1
            print "base default of 2 (binary) is used"

        base = 2
        destHistoryEmbedLength = 1
#        sourceHistoryEmbeddingLength = None  # not used at the moment
        teCalc = teCalcClass(base, destHistoryEmbedLength)

    return teCalc


def calc_infodynamics_te(infodynamicsloc, normalize, calcmethod,
                         affected_data, causal_data, test_significance=False,
                         significance_permutations=30, **parameters):
    """Calculates the transfer entropy for a specific timelag (equal to
    prediction horison) between two sets of time series data.

    This implementation makes use of the infodynamics toolkit:
    https://code.google.com/p/information-dynamics-toolkit/

    The transfer entropy should have a maximum value when timelag = delay
    used to generate an autoregressive dataset, or will otherwise indicate the
    dead time between data indicating a causal relationship.

    """

    teCalc = setup_infodynamics_te(infodynamicsloc, normalize, calcmethod,
                                   **parameters)

    sourceArray = causal_data.tolist()
    destArray = affected_data.tolist()

    sourceArrayJava = jpype.JArray(jpype.JDouble, 1)(sourceArray)
    destArrayJava = jpype.JArray(jpype.JDouble, 1)(destArray)

    teCalc.setObservations(sourceArrayJava,
                           destArrayJava)

    transentropy = teCalc.computeAverageLocalOfObservations()

    if test_significance:
        significance = teCalc.computeSignificance(significance_permutations)
    else:
        significance = None

    # Get all important properties from used teCalc
    k_history = teCalc.getProperty("k_HISTORY")
    k_tau = teCalc.getProperty("k_TAU")
    l_history = teCalc.getProperty("l_HISTORY")
    l_tau = teCalc.getProperty("l_TAU")
    delay = teCalc.getProperty("DELAY")

    properties = [k_history, k_tau, l_history, l_tau, delay]

    return transentropy, [significance, properties]


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
