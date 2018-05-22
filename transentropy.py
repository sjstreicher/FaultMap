# -*- coding: utf-8 -*-
"""Methods used in the calculation of transfer entropy. A JIDT wrapper.

"""

import jpype
import numpy as np


def setup_infodynamics_te(infodynamicsloc, calcmethod, **parameters):
    """Prepares the teCalc class of the Java Infodyamics Toolkit (JIDT)
    in order to calculate transfer entropy according to the kernel or Kraskov
    estimator method. Also supports discrete transfer entropy calculation.

    The embedding dimension of the destination or target variable (k) can
    easily be set by adjusting the histlength parameter.

    # UPDATE: (Still needs to be verified) The continous Kraskov estimator
    with the Ragwitz criterion is the basis for all results unless mentioned
    otherwise as this is the theoretical best automated procedure that is
    available in JIDT 1.3.
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

        # Normalisation is performed before this step, set property to false to
        # prevent accidental data standardisation
        teCalc.setProperty("NORMALISE", "false")

        # Parameter definitions - refer to JIDT Javadocs
        # k - destination embedded history length (Schreiber k=1)
        # kernelWidth - if NORMALISE_PROP_NAME property has been set,
        # then this kernel width corresponds to the number of
        # standard deviations from the mean (otherwise it is an absolute value)

        k = parameters.get('k', 1)
        kernel_width = parameters.get('kernel_width', 0.25)

        teCalc.initialise(k, kernel_width)

    elif calcmethod == 'kraskov':
        """The Kraskov method is the recommended method and also provides
        methods for auto-embedding. The Ragwitz criterion auto-embedding method
        will be enabled as the default.

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

        # Normalisation is performed before this step, set property to false to
        # prevent accidental data standardisation
        teCalc.setProperty("NORMALISE", "false")

        if 'auto_embed' in parameters:
            auto_embed = parameters['auto_embed']
            if auto_embed is True:
                # Enable the Ragwitz criterion
                # Enable source as well as destination embedding due to the
                # nature of our data.
                # Use a maximum history and tau search of 5
                teCalc.setProperty("AUTO_EMBED_METHOD", "RAGWITZ")

                ksearchmax = parameters.get('k_search_max', 5)
                teCalc.setProperty("AUTO_EMBED_K_SEARCH_MAX",
                                   str(ksearchmax))
                tausearchmax = parameters.get('tau_search_max', 5)
                teCalc.setProperty("AUTO_EMBED_TAU_SEARCH_MAX",
                                   str(tausearchmax))

        # Note: If setting the delay is needed to be changed on each iteration,
        # it may be best to do this outside the loop and initialise teCalc
        # after each change.

        if 'delay' in parameters:
            delay = parameters['delay']
            teCalc.setProperty("DELAY", str(delay))

        # Allow for manual override

        if 'k_history' in parameters:
            k_history = parameters['k_history']
            teCalc.setProperty("k_HISTORY", str(k_history))

        if 'k_tau' in parameters:
            k_tau = parameters['k_tau']
            teCalc.setProperty("k_TAU", str(k_tau))

        if 'l_history' in parameters:
            l_history = parameters['l_history']
            teCalc.setProperty("l_HISTORY", str(l_history))

        if 'l_tau' in parameters:
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

        base = parameters.get('base', 2)
#            print "base default of 2 (binary) is used"

        destHistoryEmbedLength = parameters.get('destHistoryEmbedLength', 1)

        base = 2
        destHistoryEmbedLength = 1
#        sourceHistoryEmbeddingLength = None  # not used at the moment
        teCalc = teCalcClass(base, destHistoryEmbedLength)
        teCalc.initialise()
    else:
        raise NameError("Transfer entropy method name not recognized")

    return teCalc


def calc_infodynamics_te(infodynamicsloc, calcmethod,
                         affected_data, causal_data, **parameters):
    """Calculates the transfer entropy for a specific timelag (equal to
    prediction horison) between two sets of time series data.

    This implementation makes use of the infodynamics toolkit:
    https://code.google.com/p/information-dynamics-toolkit/

    The transfer entropy should have a maximum value when timelag = delay
    used to generate an autoregressive dataset, or will otherwise indicate the
    dead time between data indicating a causal relationship.

    """

    teCalc = setup_infodynamics_te(infodynamicsloc, calcmethod,
                                   **parameters)

    test_significance = parameters.get('test_signifiance', False)
    significance_permutations = parameters.get('significance_permutations', 30)

#    sourceArray = causal_data.tolist()
#    destArray = affected_data.tolist()

    if (len(causal_data) != len(affected_data)):
        print("Source length: " + str(len(causal_data)))
        print("Destination length: " + str(len(affected_data)))
        raise ValueError(
            "The source and destination arrays are of different lengths")

    #sourceArrayJava = jpype.JArray(jpype.JDouble, 1)(sourceArray)
    #destArrayJava = jpype.JArray(jpype.JDouble, 1)(destArray)

#    sourceArrayJava = np.asarray(sourceArray)
#    destArrayJava = np.asarray(destArray)

    if calcmethod == 'discrete':
        source = map(int, causal_data)
        dest = map(int, affected_data)
        teCalc.addObservations(source, dest)
    else:
        teCalc.setObservations(causal_data, affected_data)

    transentropy = teCalc.computeAverageLocalOfObservations()

    # Convert nats to bits if necessary
    if calcmethod == 'kraskov':
        transentropy = transentropy / np.log(2.)
    elif calcmethod == 'kernel':
        transentropy = transentropy
    elif calcmethod == 'discrete':
        transentropy = transentropy
    else:
        raise NameError("Transfer entropy method name not recognized")

    if test_significance:
        significance = teCalc.computeSignificance(significance_permutations)
    else:
        significance = None

    # Get all important properties from used teCalc
    if calcmethod != 'discrete':
        k_history = teCalc.getProperty("k_HISTORY")
        k_tau = teCalc.getProperty("k_TAU")
        l_history = teCalc.getProperty("l_HISTORY")
        l_tau = teCalc.getProperty("l_TAU")
        delay = teCalc.getProperty("DELAY")

        properties = [k_history, k_tau, l_history, l_tau, delay]
    else:
        properties = [None]

    return transentropy, [significance, properties]


def setup_infodynamics_entropy(infodynamicsloc, estimator='kernel',
                               kernel_bandwidth=0.1, mult=False):
    """Prepares the entropyCalc class of the Java Infodyamics Toolkit (JIDK)
    in order to calculate differential entropy (continuous signals) according
    to the estimation method specified.

    Parameters
    ----------
        infodynamicsloc : path
            Location of infodynamics.jar
         estimator : string, default='kernel'
            Either 'kernel' or 'gaussian'. Specifies the estimator to use in
            determining the required probability density functions.
        kernel_bandwidth : float
            The width of the kernels for the kernel method. If normalisation
            is performed, these are in terms of standard deviation, otherwise
            absolute.
        mult : bool, default=False
            Indicates whether the entropy is to be calculated on a univariate
            or multivariate signal.

    Returns
    -------
        entropyCalc : EntropyCalculator JIDT object

    """
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(),
                       "-Xms32M",
                       "-Xmx512M",
                       "-ea",
                       "-Djava.class.path=" + infodynamicsloc)

    if estimator == 'kernel':
        if mult:
            entropyCalcClass = \
                jpype.JPackage("infodynamics.measures.continuous.kernel") \
                .EntropyCalculatorMultiVariateKernel
        else:
            entropyCalcClass = \
                jpype.JPackage("infodynamics.measures.continuous.kernel") \
                .EntropyCalculatorKernel

        entropyCalc = entropyCalcClass()
        # Normalisation is performed before this step, set property to false to
        # prevent accidental data standardisation
        entropyCalc.setProperty("NORMALISE", "false")
        entropyCalc.initialise(kernel_bandwidth)

    elif estimator == 'gaussian':
        if mult:
            entropyCalcClass = \
                jpype.JPackage("infodynamics.measures.continuous.gaussian") \
                .EntropyCalculatorMultiVariateGaussian
        else:
            entropyCalcClass = \
                jpype.JPackage("infodynamics.measures.continuous.gaussian") \
                .EntropyCalculatorGaussian

        entropyCalc = entropyCalcClass()
        entropyCalc.initialise()

    elif estimator == 'kozachenko':
        entropyCalcClass = \
            jpype.JPackage("infodynamics.measures.continuous.kozachenko") \
            .EntropyCalculatorMultiVariateKozachenko

        entropyCalc = entropyCalcClass()
        entropyCalc.initialise()

    else:
        raise NameError("Estimator not recognized")

    return entropyCalc, estimator


def calc_infodynamics_entropy(entropyCalc, data, estimator):
    """Estimates the entropy of a single signal.

    Parameters
    ----------
        entropyCalc : EntropyCalculator JIDT object
           The estimation method is determined during initialisation of this
           object beforehand.
        data : one-dimensional numpy.ndarray
           The univariate signal.

    Returns
    -------
        entropy : float
            The entropy of the signal.

    Notes
    -----
        The entropy calculated with the Gaussian estimator is in nats, while
        that calculated by the kernel estimator is in bits.
        Nats can be converted to bits by division with ln(2).
    """

    dataArray = data.tolist()
    dataArrayJava = jpype.JArray(jpype.JDouble, 1)(dataArray)
    entropyCalc.setObservations(dataArrayJava)
    entropy = entropyCalc.computeAverageLocalOfObservations()
    if estimator == 'gaussian':
        # Convert nats to bits
        entropy = entropy / np.log(2.)
    elif estimator == 'kernel':
        # Do nothing
        entropy = entropy
    elif estimator == 'kozachenko':
        entropy = entropy / np.log(2.)
    else:
        raise NameError("Estimator not recognized")

    return entropy
