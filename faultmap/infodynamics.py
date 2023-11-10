"""Methods used in the calculation of transfer entropy. A JIDT wrapper.

"""

from pathlib import Path
from typing import Literal

import jpype  # type: ignore
import numpy as np

MutualInformationMethods = Literal["kernel", "kraskov", "discrete"]
EntropyMethods = Literal["gaussian", "kernel", "kozachenko"]


def check_jvm(infodynamics_path: Path):
    """
    Check if the Java Virtual Machine (JVM) is started and start it if it is not.

    Args:
        infodynamics_path (str): The file path to the infodynamics jar file.

    Returns:
        None
    """
    if not jpype.isJVMStarted():
        jpype.startJVM(
            jpype.getDefaultJVMPath(),
            "-Xms32M",
            "-Xmx512M",
            "-ea",
            "-Djava.class.path=" + infodynamics_path.as_uri(),
            convertStrings=True,
        )


def setup_te(infodynamics_path: Path, method: MutualInformationMethods, **parameters):
    """Prepares the teCalc class of the Java Infodynamics Toolkit (JIDT)
    in order to calculate transfer entropy according to the kernel or Kraskov
    estimator method. Also supports discrete transfer entropy calculation.

    """

    check_jvm(infodynamics_path)

    if method == "kernel":
        calculator_class = jpype.JPackage(
            "infodynamics.measures.continuous.kernel"
        ).TransferEntropyCalculatorKernel
        te_calculator = calculator_class()

        # Normalisation is performed before this step, set property to false to
        # prevent accidental data standardisation
        te_calculator.setProperty("NORMALISE", "false")

        # Parameter definitions - refer to JIDT Javadocs
        # k - destination embedded history length (Schreiber k=1)
        # kernelWidth - if NORMALISE_PROP_NAME property has been set,
        # then this kernel width corresponds to the number of
        # standard deviations from the mean (otherwise it is an absolute value)

        k = parameters.get("k", 1)
        kernel_width = parameters.get("kernel_width", 0.25)

        te_calculator.initialise(k, kernel_width)

    elif method == "kraskov":
        # The Kraskov method is the recommended method and also provides methods for
        # auto-embedding. The max corr AIS auto-embedding method will be enabled as the
        # default.

        calculator_class = jpype.JPackage(
            "infodynamics.measures.continuous.kraskov"
        ).TransferEntropyCalculatorKraskov
        te_calculator = calculator_class()
        # Parameter definitions - refer to JIDT javadocs

        # k - embedding length of destination past history to consider
        # k_tau - embedding delay for the destination variable
        # l - embedding length of source past history to consider
        # l_tau - embedding delay for the source variable
        # delay - time lag between last element of source and destination
        # next value

        # Normalisation is performed before this step, set property to false to
        # prevent accidental data standardisation
        te_calculator.setProperty("NORMALISE", "false")

        # Allow for manual override

        if "k_history" in parameters:
            k_history = parameters["k_history"]
            te_calculator.setProperty("k_HISTORY", str(k_history))

        if "k_tau" in parameters:
            k_tau = parameters["k_tau"]
            te_calculator.setProperty("k_TAU", str(k_tau))

        if "l_history" in parameters:
            l_history = parameters["l_history"]
            te_calculator.setProperty("l_HISTORY", str(l_history))

        if "l_tau" in parameters:
            l_tau = parameters["l_tau"]
            te_calculator.setProperty("l_TAU", str(l_tau))

        if "auto_embed" in parameters:
            auto_embed = parameters["auto_embed"]
            if auto_embed is True:
                # Enable the Ragwitz or max AIS method criterion
                # TODO: Enable flag to determine which one
                # Enable source as well as destination embedding due to the
                # nature of our data.
                # Use a maximum history and tau search of 5
                # teCalc.setProperty("AUTO_EMBED_METHOD", "RAGWITZ")
                te_calculator.setProperty("AUTO_EMBED_METHOD", "MAX_CORR_AIS")

                ksearchmax = parameters.get("k_search_max", 5)
                te_calculator.setProperty("AUTO_EMBED_K_SEARCH_MAX", str(ksearchmax))
                tausearchmax = parameters.get("tau_search_max", 5)
                te_calculator.setProperty(
                    "AUTO_EMBED_TAU_SEARCH_MAX", str(tausearchmax)
                )

        # Note: If setting the delay is needed to be changed on each iteration,
        # it may be best to do this outside the loop and initialise teCalc
        # after each change.

        if "delay" in parameters:
            delay = parameters["delay"]
            te_calculator.setProperty("DELAY", str(delay))

        if "use_gpu" in parameters:
            if parameters["use_gpu"]:
                te_calculator.setProperty("USE_GPU", "true")
            else:
                te_calculator.setProperty("USE_GPU", "false")

        te_calculator.initialise()

    elif method == "discrete":
        calculator_class = jpype.JPackage(
            "infodynamics.measures.discrete"
        ).TransferEntropyCalculatorDiscrete
        # Parameter definitions - refer to JIDT javadocs
        # base - number of quantisation levels for each variable
        # binary variables are in base-2
        # destHistoryEmbedLength - embedded history length of the destination to
        # condition on - this is k in Schreiber's notation
        # sourceHistoryEmbeddingLength - embedded history length of the source
        # to include - this is l in Schreiber's notation
        # TODO: Allow these settings to be defined by configuration file

        # Base default of 2 (binary) is used"
        base = parameters.get("base", 2)
        dest_history_embed_length = parameters.get("destHistoryEmbedLength", 1)
        # source_history_embed_length = None  # not used at the moment

        te_calculator = calculator_class(base, dest_history_embed_length)
        te_calculator.initialise()

    else:
        raise NameError("Transfer entropy method name not recognized")

    return te_calculator


def calc_te(infodynamics_path, calc_method, affected_data, causal_data, **parameters):
    """Calculates the transfer entropy for a specific time lag (equal to
    prediction horizon) between two sets of time series data.

    This implementation makes use of the infodynamics toolkit:
    https://jlizier.github.io/jidt/

    The transfer entropy should have a maximum value when time lag = delay
    used to generate an autoregressive dataset, or will otherwise indicate the
    dead time between data indicating a causal relationship.

    """

    te_calc = setup_te(infodynamics_path, calc_method, **parameters)
    mi_calc = setup_mi(infodynamics_path, calc_method, **parameters)

    test_significance = parameters.get("test_significance", False)
    significance_permutations = parameters.get("significance_permutations", 30)

    if len(causal_data) != len(affected_data):
        print("Source length: " + str(len(causal_data)))
        print("Destination length: " + str(len(affected_data)))
        raise ValueError("The source and destination arrays are of different lengths")

    if calc_method == "discrete":
        source = map(int, causal_data)
        destination = map(int, affected_data)
        te_calc.addObservations(source, destination)
        mi_calc.addObservations(source, destination)
    else:
        te_calc.setObservations(causal_data, affected_data)
        mi_calc.setObservations(causal_data, affected_data)

    transfer_entropy = te_calc.computeAverageLocalOfObservations()
    mutual_information = mi_calc.computeAverageLocalOfObservations()

    # Convert nats to bits if necessary
    if calc_method == "kraskov":
        transfer_entropy = transfer_entropy / np.log(2.0)
        mutual_information = mutual_information / np.log(2.0)
    elif calc_method in ("kernel", "discrete"):
        pass
    else:
        raise NameError("Infodynamics method name not recognized")

    if test_significance:
        te_significance = te_calc.computeSignificance(significance_permutations)
        mi_significance = mi_calc.computeSignificance(significance_permutations)
    else:
        te_significance = None
        mi_significance = None

    # Get all important properties from used te_calc
    if calc_method != "discrete":
        k_history = te_calc.getProperty("k_HISTORY")
        k_tau = te_calc.getProperty("k_TAU")
        l_history = te_calc.getProperty("l_HISTORY")
        l_tau = te_calc.getProperty("l_TAU")
        delay = te_calc.getProperty("DELAY")

        properties = [
            k_history,
            k_tau,
            l_history,
            l_tau,
            delay,
        ]
    else:
        properties = [None]

    return (
        transfer_entropy,
        [[te_significance, mi_significance], properties, mutual_information],
    )


def setup_mi(infodynamics_path: Path, method: str, **parameters):
    """Prepares the mi_calc class of the Java Infodynamics Toolkit (JIDT)
    in order to calculate mutual information according to the kernel or Kraskov
    estimator method. Also supports discrete mutual information calculation.

    The Kraskov method is the recommended method and also provides
    methods for auto-embedding. The max corr AIS auto-embedding method
    will be enabled as the default.

    """

    check_jvm(infodynamics_path)

    if method == "kernel":
        mi_calc_class = jpype.JPackage(
            "infodynamics.measures.continuous.kernel"
        ).MutualInfoCalculatorMultiVariateKernel
        mi_calc = mi_calc_class()

        # Normalisation is performed before this step, set property false to
        # prevent accidental data standardisation
        mi_calc.setProperty("NORMALISE", "false")

        # Parameter definitions - refer to JIDT Javadocs
        # k - destination embedded history length (Schreiber k=1)
        # kernelWidth - if NORMALISE_PROP_NAME property has been set,
        # then this kernel width corresponds to the number of
        # standard deviations from the mean (otherwise it is an absolute value)

        # k = parameters.get("k", 1)
        kernel_width = parameters.get("kernel_width", 0.25)

        mi_calc.setProperty("KERNEL_WIDTH", str(kernel_width))

        mi_calc.initialise()

    elif method == "kraskov":
        mi_calc_class = jpype.JPackage(
            "infodynamics.measures.continuous.kraskov"
        ).MutualInfoCalculatorMultiVariateKraskov1
        mi_calc = mi_calc_class()
        # Parameter definitions - refer to JIDT javadocs

        # k - embedding length of destination past history to consider
        # delay - time lag between last element of source and destination
        # next value

        # Normalisation is performed before this step, set property false to
        # prevent accidental data standardisation
        mi_calc.setProperty("NORMALISE", "false")

        # Note: If setting the delay is needed to be changed on each iteration,
        # it may be best to do this outside the loop and initialise teCalc
        # after each change.

        if "delay" in parameters:
            delay = parameters["delay"]
            mi_calc.setProperty("TIME_DIFF", str(delay))

        mi_calc.initialise()

    elif method == "discrete":
        mi_calc_class = jpype.JPackage(
            "infodynamics.measures.discrete"
        ).MutualInformationCalculatorDiscrete
        # Parameter definitions - refer to JIDT javadocs
        # base - number of quantisation levels for each variable
        # binary variables are in base 2
        # TODO: Allow these settings to be defined by configuration file

        base = parameters.get("base", 2)
        #            print "base default of 2 (binary) is used"

        mi_calc = mi_calc_class(base, base, 0)
        mi_calc.initialise()

    else:
        raise NameError("Mutual information method name not recognized")

    return mi_calc


def setup_entropy(
    infodynamics_path,
    estimator=EntropyMethods,
    kernel_bandwidth=0.1,
    multivariate=False,
):
    """Prepares the entropy_calc class of the Java Infodynamics Toolkit (JIDT)
    in order to calculate differential entropy (continuous signals) according
    to the estimation method specified.

    Parameters
    ----------
        infodynamics_path : path
            Location of infodynamics.jar
         estimator : string, default='kernel'
            Either 'kernel' or 'gaussian'. Specifies the estimator to use in
            determining the required probability density functions.
        kernel_bandwidth : float
            The width of the kernels for the kernel method. If normalisation
            is performed, these are in terms of standard deviation, otherwise
            absolute.
        multivariate : bool, default=False
            Indicates whether the entropy is to be calculated on a univariate
            or multivariate signal.

    Returns
    -------
        entropy_calc : EntropyCalculator JIDT object

    Args:
        estimator:

    """

    check_jvm(infodynamics_path)

    if estimator == "kernel":
        if multivariate:
            entropy_calc_class = jpype.JPackage(
                "infodynamics.measures.continuous.kernel"
            ).EntropyCalculatorMultiVariateKernel
        else:
            entropy_calc_class = jpype.JPackage(
                "infodynamics.measures.continuous.kernel"
            ).EntropyCalculatorKernel

        entropy_calc = entropy_calc_class()
        # Normalisation is performed before this step, set property false to
        # prevent accidental data standardisation
        entropy_calc.setProperty("NORMALISE", "false")
        entropy_calc.initialise(kernel_bandwidth)

    elif estimator == "gaussian":
        if multivariate:
            entropy_calc_class = jpype.JPackage(
                "infodynamics.measures.continuous.gaussian"
            ).EntropyCalculatorMultiVariateGaussian
        else:
            entropy_calc_class = jpype.JPackage(
                "infodynamics.measures.continuous.gaussian"
            ).EntropyCalculatorGaussian

        entropy_calc = entropy_calc_class()
        entropy_calc.initialise()

    elif estimator == "kozachenko":
        entropy_calc_class = jpype.JPackage(
            "infodynamics.measures.continuous.kozachenko"
        ).EntropyCalculatorMultiVariateKozachenko

        entropy_calc = entropy_calc_class()
        entropy_calc.initialise()

    else:
        raise NameError("Estimator not recognized")

    return entropy_calc, estimator


def calc_entropy(entropy_calculator, data, estimator: EntropyMethods) -> float:
    """Estimates the entropy of a single signal.

    Parameters
    ----------
        entropy_calculator : EntropyCalculator JIDT object
           The estimation method is determined during initialisation of this
           object beforehand.
        data : one-dimensional numpy.ndarray
           The uni-variate signal.

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

    data_array = data.tolist()
    data_array_java = jpype.JArray(jpype.JDouble, 1)(data_array)
    entropy_calculator.setObservations(data_array_java)
    entropy = entropy_calculator.computeAverageLocalOfObservations()
    if estimator == "gaussian":
        # Convert nats to bits
        entropy = entropy / np.log(2.0)
    elif estimator == "kernel":
        pass
    elif estimator == "kozachenko":
        entropy = entropy / np.log(2.0)
    else:
        raise NameError("Estimator not recognized")

    return entropy
