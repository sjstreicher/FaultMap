"""Module containing functions used in calculation of transfer entropy
Created on Mon Feb 24 15:18:33 2014

@author: Simon Streicher
"""
import numpy as np
from scipy import stats
import random
import mcint
from autoreg import getdata
from jpype import *


def vectorselection(data, timelag, sub_samples, k=1, l=1):
    """Generates sets of vectors for calculating transfer entropy.

    For notation references see Shu2013.

    Takes into account the time lag (number of samples between vectors of the
    same variable).

    In this application the prediction horizon (h) is set to equal
    to the time lag.

    The first vector in the data array should be the samples of the variable
    to be predicted (x) while the second vector should be sampled of the vector
    used to make the prediction (y).

    sub_samples is the amount of samples in the dataset used to calculate the
    transfer entropy between two vectors.
    The required number of samples is extracted from the end of the vector.
    If the vector is longer than the number of samples specified plus the
    desired time lag then the remained of the data will be discarded.
    sub_samples <= samples


    k refers to the dimension of the historical data to be predicted (x)

    l refers to the dimension of the historical data used
    to do the prediction (y)

    """
    _, sample_n = data.shape
    x_pred = data[0, sample_n-sub_samples-1:-1]

    x_hist = np.zeros((k, sub_samples))
    y_hist = np.zeros((l, sub_samples))

    for n in range(1, (k+1)):
        # Original form according to Bauer (2007)
#        x_hist[n-1, :] = data[0, ((sample_n - samples) - timelag * n):
#                               (sample_n - timelag * n)]
        # Modified form according to Shu & Zhao (2013)
        x_hist[n-1, :] = data[0, ((sample_n - sub_samples) - timelag *
                                  (n-1) - 2):(sample_n - timelag * (n-1) - 2)]
    for m in range(1, (l+1)):
        y_hist[m-1:, :] = data[1, ((sample_n - sub_samples) -
                               timelag * (m) - 1):
                               (sample_n - timelag * (m) - 1)]

    return x_pred, x_hist, y_hist


def pdfcalcs(x_pred, x_hist, y_hist):
    """Calculates the PDFs required to calculate transfer entropy.

    Currently only supports k = 1; l = 1

    """
    # TODO: Generalize for k and l

    # Get dimensions of vectors
#    k = np.size(x_hist[:, 1])
#    l = np.size(y_hist[:, 1])

    # Calculate p(x_{i+h}, x_i, y_i)
    data_1 = np.vstack([x_pred, x_hist[0, :], y_hist[0, :]])
    pdf_1 = stats.gaussian_kde(data_1, 'silverman')

    # Calculate p(x_i, y_i)
    data_2 = np.vstack([x_hist[0, :], y_hist[0, :]])
    pdf_2 = stats.gaussian_kde(data_2, 'silverman')

    # Calculate p(x_{i+h}, x_i)
    data_3 = np.vstack([x_pred, x_hist[0, :]])
    pdf_3 = stats.gaussian_kde(data_3, 'silverman')

    # Calculate p(x_i)
    data_4 = x_hist[0, :]
    pdf_4 = stats.gaussian_kde(data_4, 'silverman')

    return pdf_1, pdf_2, pdf_3, pdf_4


def te_elementcalc(pdf_1, pdf_2, pdf_3, pdf_4, x_pred_val,
                   x_hist_val, y_hist_val):
    """Calculate elements for summation for a specific set of coordinates"""

    # Function evaluations
    term1 = pdf_1([x_pred_val, x_hist_val, y_hist_val])
    term2 = pdf_2([x_hist_val, y_hist_val])
    term3 = pdf_3([x_pred_val, x_hist_val])
    term4 = pdf_4([x_hist_val])

    logterm_num = (term1 / term2)
    logterm_den = (term3 / term4)
    coeff = term1
    np.seterr(divide='ignore', invalid='ignore')
    sum_element = coeff * np.log10(logterm_num / logterm_den)
    np.seterr(divide=None, invalid=None)

    #print sum_element

    # TODO: Need to find a proper way to correct for cases when PDFs return
    # nan or inf values.
    # Most of the PDF issues are associated with the x_hist values being
    # very similar to the x_pred values.
    # Some very small negative values are sometimes returned.

    if (str(sum_element[0]) == 'nan' or str(sum_element[0]) == 'inf'
            or sum_element[0] < 0):
        sum_element = 0

    return sum_element


def autoreg_datagen(delay, timelag, samples, sub_samples, k=1, l=1):
    """Generates autoreg data for a specific timelag (equal to
    prediction horison) for a set of autoregressive data.

    sub_samples is the amount of samples in the dataset used to calculate the
    transfer entropy between two vectors (taken from the end of the dataset).
    sub_samples <= samples

    Currently only supports k = 1; l = 1

    You can search through a set of timelags in an attempt to identify the
    original delay.
    The transfer entropy should have a maximum value when timelag = delay
    used to generate the autoregressive dataset.

    """

    data = getdata(samples, delay)

    [x_pred, x_hist, y_hist] = vectorselection(data, timelag,
                                               sub_samples, k, l)

    return x_pred, x_hist, y_hist


def setup_infodynamics_te():
    # Change location of jar to match yours:
    jarLocation = "infodynamics.jar"
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

    teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    teCalc = teCalcClass()
    # Normalise the individual variables
    teCalc.setProperty("NORMALISE", "true")
    # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
    teCalc.initialise(1, 0.5)

    return teCalc


def calc_infodynamics_te(teCalc, x_hist, y_hist):
    """Calculates the transfer entropy for a specific timelag (equal to
    prediction horison) for a set of autoregressive data.

    This implementation makes use of the infodynamics toolkit:
    https://code.google.com/p/information-dynamics-toolkit/

    sub_samples is the amount of samples in the dataset used to calculate the
    transfer entropy between two vectors (taken from the end of the dataset).
    sub_samples <= samples

    Currently only supports k = 1; l = 1;

    You can search through a set of timelags in an attempt to identify the
    original delay.
    The transfer entropy should have a maximum value when timelag = delay
    used to generate the autoregressive dataset.

    """

    sourceArray = y_hist.tolist()
    destArray = x_hist.tolist()

    teCalc.setObservations(JArray(JDouble, 1)(sourceArray),
                           JArray(JDouble, 1)(destArray))

    transentropy = teCalc.computeAverageLocalOfObservations()
    print transentropy

    return transentropy


def calc_custom_te(x_pred, x_hist, y_hist, mcsamples):
    """Calculates the transfer entropy between two variables from a set of
    vectors already calculated.

    ampbins is the number of amplitude bins to use over each variable

    """

    # First do an example for the case of k = l = 1
    # TODO: Sum loops to allow for a general case

    # Divide the range of each variable into amplitude bins to sum over

    # TODO: Will make this general

    x_pred_min = x_pred.min()
    x_pred_max = x_pred.max()
    x_hist_min = x_hist.min()
    x_hist_max = x_hist.max()
    y_hist_min = y_hist.min()
    y_hist_max = y_hist.max()

    x_pred_range = x_pred_max - x_pred_min
    x_hist_range = x_hist_max - x_hist_min
    y_hist_range = y_hist_max - y_hist_min

    # Calculate PDFs for all combinations required
    [pdf_1, pdf_2, pdf_3, pdf_4] = pdfcalcs(x_pred, x_hist, y_hist)

    def integrand(x):
        s1 = x[0]
        s2 = x[1]
        s3 = x[2]

        return te_elementcalc(pdf_1, pdf_2, pdf_3, pdf_4,
                              s1, s2, s3)

    def sampler():
        while True:
            s1 = random.uniform(x_pred_min, x_pred_max)
            s2 = random.uniform(x_hist_min, x_hist_max)
            s3 = random.uniform(y_hist_min, y_hist_max)
            yield(s1, s2, s3)

    domainsize = x_pred_range * x_hist_range * y_hist_range
#    domainsize = 1

#    print domainsize

    # Do triple integration using scipy.integrate.tplquad
    # Also do a version using mcint
    # See this for higher orders:
    # http://stackoverflow.com/questions/14071704/integrating-a-multidimensional-integral-in-scipy

    for nmc in [mcsamples]:
        random.seed(1)
        result, error = mcint.integrate(integrand, sampler(),
                                        measure=domainsize, n=nmc)

#        print "Using n = ", nmc
        print "Result = ", result[0]
    return result
