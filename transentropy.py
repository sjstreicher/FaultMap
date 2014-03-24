"""Module containing functions used in calculation of transfer entropy
Created on Mon Feb 24 15:18:33 2014

@author: Simon Streicher
"""
import numpy as np
from scipy import stats
import jpype
from sklearn import preprocessing


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
    x_pred = x_pred[np.newaxis, :]

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


def te_local_elementcalc(pdf_1, pdf_2, pdf_3, pdf_4, x_pred_val,
                         x_hist_val, y_hist_val):
    """Calculate elements for summation for a specific set of coordinates"""

    # Function evaluations
    term1 = pdf_1([x_pred_val, x_hist_val, y_hist_val])
    term2 = pdf_2([x_hist_val, y_hist_val])
    term3 = pdf_3([x_pred_val, x_hist_val])
    term4 = pdf_4([x_hist_val])

    logterm_num = (term1 / term2)
    logterm_den = (term3 / term4)
    np.seterr(divide='ignore', invalid='ignore')
    sum_element = np.log2(logterm_num / logterm_den)
    np.seterr(divide=None, invalid=None)

    #print sum_element

    # TODO: Need to find a proper way to correct for cases when PDFs return
    # nan or inf values.
    # Most of the PDF issues are associated with the x_hist values being
    # very similar to the x_pred values.
    # Some very small negative values are sometimes returned.

    if (str(sum_element[0]) == 'nan' or str(sum_element[0]) == 'inf'):
        sum_element = 0

    return sum_element


def te_overall_elementcalc(pdf_1, pdf_2, pdf_3, pdf_4, x_pred_val,
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
    sum_element = coeff * np.log2(logterm_num / logterm_den)
    np.seterr(divide=None, invalid=None)

    # TODO: Need to find a proper way to correct for cases when PDFs return
    # nan or inf values.
    # Most of the PDF issues are associated with the x_hist values being
    # very similar to the x_pred values.
    # Some very small negative values are sometimes returned.

    if (str(sum_element[0]) == 'nan' or str(sum_element[0]) == 'inf'):
        sum_element = 0

    return sum_element


def calc_custom_local_te(x_pred, x_hist, y_hist):
    """Calculates the local transfer entropy between two variables.

    See Lizier2008 Eq. 8 for implementation reference.

    The x_pred, x_hist and y_hist vectors need to be determined externally.

    """

    # First do an example for the case of k = l = 1

    # TODO: Make this general for k and l

    # TODO: Review implementation - something is very wrong here with the scale
    # of the result

    # Normalise data
    x_pred_norm = preprocessing.scale(x_pred, axis=1)
    x_hist_norm = preprocessing.scale(x_hist, axis=1)
    y_hist_norm = preprocessing.scale(y_hist, axis=1)

    # Get the number of observations
    numobs = x_pred.shape[1]

    # Calculate PDFs for all combinations required
    [pdf_1, pdf_2, pdf_3, pdf_4] = pdfcalcs(x_pred_norm,
                                            x_hist_norm, y_hist_norm)

    tesum = 0.0
    for x_pred_val, x_hist_val, y_hist_val in zip(x_pred_norm[0],
                                                  x_hist_norm[0],
                                                  y_hist_norm[0]):

        tesum += te_local_elementcalc(pdf_1, pdf_2, pdf_3, pdf_4,
                                      x_pred_val, x_hist_val, y_hist_val)
    transent = tesum / numobs

    return transent


def calc_custom_overall_te(x_pred, x_hist, y_hist):
    """Calculates the transfer entropy between two variables.

    See Shu2013 Eq. 1 for implementation reference.

    The x_pred, x_hist and y_hist vectors need to be determined externally.

    """

    # First do an example for the case of k = l = 1

    # TODO: Make this general for k and l

    # Normalise data
    x_pred_norm = preprocessing.scale(x_pred, axis=1)
    x_hist_norm = preprocessing.scale(x_hist, axis=1)
    y_hist_norm = preprocessing.scale(y_hist, axis=1)

    # Calculate PDFs for all combinations required
    [pdf_1, pdf_2, pdf_3, pdf_4] = pdfcalcs(x_pred_norm,
                                            x_hist_norm, y_hist_norm)

    # Needs to be completed (reverted to mcint)

    return None


def setup_infodynamics_te():

    teCalcClass = jpype.JPackage("infodynamics.measures.continuous.kernel") \
                       .TransferEntropyCalculatorKernel
    teCalc = teCalcClass()
    # Normalise the individual variables
    teCalc.setProperty("NORMALISE", "true")

    return teCalc


def calc_infodynamics_te(teCalc, affected_data, causal_data):
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

    # Normalise data to be safe
    affected_data_norm = preprocessing.scale(affected_data, axis=1)
    causal_data_norm = preprocessing.scale(causal_data, axis=1)

    # Use history length 1 (Schreiber k=1),
    # kernel width of 0.5 normalised units
    teCalc.initialise(1, 0.5)

    sourceArray = causal_data_norm[0].tolist()
    destArray = affected_data_norm[0].tolist()

    teCalc.setObservations(jpype.JArray(jpype.JDouble, 1)(sourceArray),
                           jpype.JArray(jpype.JDouble, 1)(destArray))

    transentropy = teCalc.computeAverageLocalOfObservations()

    return transentropy
