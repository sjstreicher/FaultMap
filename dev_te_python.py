# -*- coding: utf-8 -*-
"""This method is an implementation of transfer entropy calculation making use
of Python libraries only.

The transfer entropy is calculated according to the global average of local
entropies method as presented in Lizier2008.

@author: Simon Streicher
"""

import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from scipy import stats

from datagen import autoreg_datagen


def vectorselection(data, timelag, sub_samples, k=1, l=1):
    """Generates sets of vectors from tags time series data
    for calculating transfer entropy.

    For notation references see Shu2013.

    Takes into account the time lag (number of samples between vectors of the
    same variable).

    In this application the prediction horizon (h) is set to equal
    to the time lag.

    The first vector in the data array should be the samples of the causal
    variable (y) (source) while the second vector should be
    sampled of the vector to be predicted (x) (destination).

    The sub_samples parameter is the amount of samples in the dataset used to
    calculate the transfer entropy between two vectors and must satisfy
    sub_samples <= samples.

    The required number of samples is extracted from the end of the vector.
    If the vector is longer than the number of samples specified plus the
    desired time lag then the remained of the data will be discarded.

    k refers to the dimension of the historical data to be predicted (x)
    (affected data)

    l refers to the dimension of the historical data used
    to do the prediction (y)
    (causal data)

    """
    _, sample_n = data.shape
    x_pred = data[1, sample_n-sub_samples:]
    x_pred = x_pred[np.newaxis, :]

    x_hist = np.zeros((k, sub_samples))
    y_hist = np.zeros((l, sub_samples))

    for n in range(1, (k+1)):
        # Original form according to Bauer (2007)
        # TODO: Provide for comparison
        # Modified form according to Shu & Zhao (2013)
        x_hist[n-1, :] = data[1, ((sample_n - sub_samples) - timelag):
                              (sample_n - timelag)]
    for m in range(1, (l+1)):
        y_hist[m-1:, :] = data[0, ((sample_n - sub_samples) - timelag):
                               (sample_n-timelag)]

    return x_pred[0], x_hist[0], y_hist[0]


def theta_calc_shu(data, variables):
    C = (4.0/3.0)**(1.0/5.0)
    sigma = np.std(data)
    N = len(data)
    theta = C * sigma * (N**(-1.0/(4.0 + variables)))
    return theta


def kernel_calc_shu(data, sample, value, variables):
    """Calculates a single single-variable Gaussian Kernel.

    data is the single variable set of sample points
    sample is the element of the data set used in the
    calculation of the single kernel

    value is the point according to which the probability density function
    is evaluated

    variables is the number of variables involved in the greater
    multivariable PDF calculation of which this kernel will form a part

    """

    theta = theta_calc_shu(data, variables)
    kernel = ((1.0 / (np.sqrt(2*np.pi) * theta)) *
              np.exp(-(value - sample)**2 / (2 * (theta**2))))
    return kernel


def joint_pdf_eval_shu(data, values):
    """Evaluates the joint (multivariable) probability density function
    for a specific set of values.

    data is the multidimensional array containing all samples for all variables
    values is the set of values around which the joint PDF is to be evaluated

    """

    if (np.shape(data)[0] > np.shape(data)[1]):
        # This catches the case of a single variable
        (numsamples, numvars) = np.shape(data)
    else:
        (numvars, numsamples) = np.shape(data)

    # TODO: Test that samples and number of variable in data
    #       has the same dimension.
    # TODO: Implement amplitude bins in order to select points
    #       to use as samples
    #       First write for the case where all variables
    #       use the same number of ampbins
    #       Follow the implementation of Bauer for calculating amplitude bins
    kernel_sum = 0
    #samples = np.zeros([numvars, ampbins])
    #mins = np.zeros([1, numvars])
    #maxs = np.zeros([1, numvars])
    for i in range(numsamples):
        kernel_prod = 1
        if numvars == 1:
            kernel_prod = kernel_prod * kernel_calc_shu(data, data[i],
                                                        values, numvars)
        elif (numvars > 1):
            for n in range(numvars):
                #mins[n] = data[n].min()
                #maxs[n] = data[n].max()
                #samples[n] = np.linsapce(mins[n], maxs[n], ampbins)
                kernel_prod = kernel_prod * \
                    kernel_calc_shu(data[n], data[n][i], values[n], numvars)

        kernel_sum += kernel_prod

    prob = (1.0 / numsamples) * kernel_sum
    return prob


def pdfcalcs_custom_shu(x_pred, x_hist, y_hist, x_pred_val,
                        x_hist_val, y_hist_val):
    """Evaluates the PDFs required to calculate transfer entropy.

    Currently only supports k = 1; l = 1

    """
    # TODO: Generalize for k and l

    # Get dimensions of vectors
#    k = np.size(x_hist[:, 1])
#    l = np.size(y_hist[:, 1])

    # Calculate p(x_{i+h}, x_i, y_i)
    data_1 = np.vstack([x_pred, x_hist, y_hist])
    pdf_1 = joint_pdf_eval_shu(data_1, [x_pred_val, x_hist_val, y_hist_val])
    #pdf_1 = stats.gaussian_kde(data_1, 'silverman')

    # Calculate p(x_i, y_i)
    data_2 = np.vstack([x_hist, y_hist])
    pdf_2 = joint_pdf_eval_shu(data_2, [x_hist_val, y_hist_val])
    #pdf_2 = stats.gaussian_kde(data_2, 'silverman')

    # Calculate p(x_{i+h}, x_i)
    data_3 = np.vstack([x_pred, x_hist])
    pdf_3 = joint_pdf_eval_shu(data_3, [x_pred_val, x_hist_val])
    #pdf_3 = stats.gaussian_kde(data_3, 'silverman')

    # Calculate p(x_i)
    data_4 = np.vstack(x_hist)
    pdf_4 = joint_pdf_eval_shu(data_4, [x_hist_val])
    #pdf_4 = stats.gaussian_kde(data_4, 'silverman')

    return pdf_1, pdf_2, pdf_3, pdf_4


def pdfcalcs_scipy(x_pred, x_hist, y_hist):
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


def te_eq8_elementcalc(pdf_1, pdf_2, pdf_3, pdf_4, x_pred_val,
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


def te_eq4_elementcalc(pdf_1, pdf_2, pdf_3, pdf_4, x_pred_val,
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


def te_shu_elementcalc(x_pred, x_hist, y_hist, x_pred_val,
                       x_hist_val, y_hist_val):
    """Calculate elements for summation for a specific set of coordinates"""

    # Need to find a proper way to correct for cases when PDFs return 0
    # Most of the PDF issues are associated with the x_hist values being
    # very similar to the x_pred values
    # Some very small negative values are sometimes returned

    # Function evaluations

    [term1, term2, term3, term4] = \
        pdfcalcs_custom_shu(x_pred, x_hist, y_hist, x_pred_val,
                            x_hist_val, y_hist_val)

    logterm_num = (term1 / term2)
    logterm_den = (term3 / term4)
    coeff = term1
    sum_element = coeff * np.log(logterm_num / logterm_den)

    # TODO: This still needs to be justified
    if (str(sum_element[0]) == 'nan' or str(sum_element[0]) == 'inf'):
        sum_element = 0

    return sum_element


def calc_custom_eq8_te(x_pred, x_hist, y_hist):
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

        sumelement = te_eq8_elementcalc(pdf_1, pdf_2, pdf_3, pdf_4,
                                        x_pred_val, x_hist_val, y_hist_val)
        tesum += sumelement
    transent = tesum / numobs

    return transent


def calc_custom_eq4_te(x_pred, x_hist, y_hist, mcsamples):
    """Calculates the transfer entropy between two variables.

    See Lizier2008 Eq. 4 for implementation reference.

    The x_pred, x_hist and y_hist vectors need to be determined externally.

    """

    # First do an example for the case of k = l = 1

    # TODO: Make this general for k and l

    # Normalise data
    x_pred_norm = preprocessing.scale(x_pred, axis=1)
    x_hist_norm = preprocessing.scale(x_hist, axis=1)
    y_hist_norm = preprocessing.scale(y_hist, axis=1)

    x_pred_min = x_pred_norm.min()
    x_pred_max = x_pred_norm.max()
    x_hist_min = x_hist_norm.min()
    x_hist_max = x_hist_norm.max()
    y_hist_min = y_hist_norm.min()
    y_hist_max = y_hist_norm.max()

    x_pred_range = x_pred_max - x_pred_min
    x_hist_range = x_hist_max - x_hist_min
    y_hist_range = y_hist_max - y_hist_min

    # Calculate PDFs for all combinations required
    [pdf_1, pdf_2, pdf_3, pdf_4] = pdfcalcs(x_pred_norm,
                                            x_hist_norm, y_hist_norm)

    def integrand(x):
        s1 = x[0]
        s2 = x[1]
        s3 = x[2]

        return te_eq4_elementcalc(pdf_1, pdf_2, pdf_3, pdf_4,
                                  s1, s2, s3)

    def sampler():
        while True:
            s1 = random.uniform(x_pred_min, x_pred_max)
            s2 = random.uniform(x_hist_min, x_hist_max)
            s3 = random.uniform(y_hist_min, y_hist_max)
            yield(s1, s2, s3)

    domainsize = x_pred_range * x_hist_range * y_hist_range

    for nmc in [mcsamples]:
        random.seed(1)
        result, error = mcint.integrate(integrand, sampler(),
                                        measure=domainsize, n=nmc)

    return result



def kde_sklearn_create(x, bandwidth=0.2, autobandwidth=False, **kwargs):
    """Kernel Density Estimation with Scikit-learn creation"""
    if autobandwidth:
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': np.linspace(0.4, 0.5, 100)},
                            cv=20)  # 20-fold cross-validation
        grid.fit(x)
        kde_skl = grid.best_estimator_
        print grid.best_params_
    else:
        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(x)
    return kde_skl


def kde_sklearn_sample(kde_skl, kde_grid):
    log_pdf = kde_skl.score_samples(kde_grid[:, np.newaxis])
    return np.exp(log_pdf)


def sklearn_pdfcalcs(x_pred, x_hist, y_hist):
    """Calculates the PDFs (as kernel density enstimates)
    required to calculate transfer entropy.

    Currently only supports k = 1; l = 1

    """
    # TODO: Generalize for k and l

    # Get dimensions of vectors
#    k = np.size(x_hist[:, 1])
#    l = np.size(y_hist[:, 1])

    # Estimate p(x_{i+h}, x_i, y_i)
    data_1 = np.vstack([x_pred, x_hist, y_hist])
    pdf_1 = kde_sklearn_create(data_1.T)
    pdf_1_read = kde_sklearn_sample(pdf_1, data_1.T)

    # Estimate p(x_i, y_i)
    data_2 = np.vstack([x_hist, y_hist])
    pdf_2 = kde_sklearn_create(data_2.T)
    pdf_2_read = kde_sklearn_sample(pdf_2, data_2.T)

    # Calculate p(x_{i+h}, x_i)
    data_3 = np.vstack([x_pred, x_hist])
    pdf_3 = kde_sklearn_create(data_3.T)
    pdf_3_read = kde_sklearn_sample(pdf_3, data_3.T)

    # Calculate p(x_i)
    data_4 = np.vstack([x_hist])
    pdf_4 = kde_sklearn_create(data_4.T)
    pdf_4_read = kde_sklearn_sample(pdf_4, data_4.T)

    return pdf_1_read, pdf_2_read, pdf_3_read, pdf_4_read


def scipy_pdfcalcs(x_pred, x_hist, y_hist):
    """Calculates the PDFs (as kernel density enstimates)
    required to calculate transfer entropy.

    Currently only supports k = 1; l = 1

    """
    # TODO: Generalize for k and l

    # Get dimensions of vectors
#    k = np.size(x_hist[:, 1])
#    l = np.size(y_hist[:, 1])

    # Calculate p(x_{i+h}, x_i, y_i)
    data_1 = np.vstack([x_pred, x_hist, y_hist])
    pdf_1 = stats.gaussian_kde(data_1, 'silverman')

    # Calculate p(x_i, y_i)
    data_2 = np.vstack([x_hist, y_hist])
    pdf_2 = stats.gaussian_kde(data_2, 'silverman')

    # Calculate p(x_{i+h}, x_i)
    data_3 = np.vstack([x_pred, x_hist])
    pdf_3 = stats.gaussian_kde(data_3, 'silverman')

    # Calculate p(x_i)
    data_4 = np.vstack([x_hist])
    pdf_4 = stats.gaussian_kde(data_4, 'silverman')

    return pdf_1, pdf_2, pdf_3, pdf_4


def te_calc_sklearn(pdf_1, pdf_2, pdf_3, pdf_4):
    """Calculate elements for summation for a specific set of coordinates"""

    logterm_num = (pdf_1 / pdf_2)
    logterm_den = (pdf_3 / pdf_4)
#    np.seterr(divide='ignore', invalid='ignore')
    sum_elements = np.sum(np.log2(logterm_num / logterm_den))
#    np.seterr(divide=None, invalid=None)
    tentropy = (sum_elements / len(pdf_1))
    return tentropy


def te_calc_scipy(pdf_1, pdf_2, pdf_3, pdf_4):
    """Calculate elements for summation for a specific set of coordinates"""

    logterm_num = (pdf_1 / pdf_2)
    logterm_den = (pdf_3 / pdf_4)
#    np.seterr(divide='ignore', invalid='ignore')
    sum_elements = np.sum(np.log2(logterm_num / logterm_den))
#    np.seterr(divide=None, invalid=None)
    tentropy = (sum_elements / len(pdf_1))
    return tentropy

def te_eq8_elementcalc(pdf_1, pdf_2, pdf_3, pdf_4, x_pred_val,
                       x_hist_val, y_hist_val):
    """Calculate elements for summation for a specific set of coordinates"""

    # Function evaluations
    term1 = pdf_1([x_pred_val, x_hist_val, y_hist_val])
    term2 = pdf_2([x_hist_val, y_hist_val])
    term3 = pdf_3([x_pred_val, x_hist_val])
    term4 = pdf_4([x_hist_val])

    logterm_num = (term1 / term2)
    logterm_den = (term3 / term4)
#    np.seterr(divide='ignore', invalid='ignore')
    sum_element = np.log2(logterm_num / logterm_den)
#    np.seterr(divide=None, invalid=None)

    #print sum_element

    # TODO: Need to find a proper way to correct for cases when PDFs return
    # nan or inf values.
    # Most of the PDF issues are associated with the x_hist values being
    # very similar to the x_pred values.
    # Some very small negative values are sometimes returned.

    if (str(sum_element[0]) == 'nan' or str(sum_element[0]) == 'inf'):
        sum_element = 0



    return sum_element


def calc_scipy_te(x_pred, x_hist, y_hist):
    """Calculates the local transfer entropy between two variables.

    See Lizier2008 Eq. 8 for implementation reference.

    The x_pred, x_hist and y_hist vectors need to be determined externally.

    Makes use of the scipy Gaussian Kernel estimation method

    """

    # First do an example for the case of k = l = 1

    # TODO: Make this general for k and l

    # TODO: Review implementation - something is very wrong here with the scale
    # of the result

    # Get the number of observations
    numobs = len(x_pred)

    # Calculate PDFs for all combinations required
    [pdf_1, pdf_2, pdf_3, pdf_4] = scipy_pdfcalcs(x_pred,
                                                  x_hist, y_hist)

    tesum = 0.0
    for x_pred_val, x_hist_val, y_hist_val in zip(x_pred,
                                                  x_hist,
                                                  y_hist):

        sumelement = te_eq8_elementcalc(pdf_1, pdf_2, pdf_3, pdf_4,
                                        x_pred_val, x_hist_val, y_hist_val)
        tesum += sumelement
    transent = tesum / numobs

    return transent



def calc_sklearn_te(x_pred, x_hist, y_hist):
    [pdf_1, pdf_2, pdf_3, pdf_4] = sklearn_pdfcalcs(x_pred, x_hist, y_hist)
    tentropy = te_calc_sklearn(pdf_1, pdf_2, pdf_3, pdf_4)
    return tentropy


def calc_custom_shu_te(x_pred, x_hist, y_hist, approach='averageoflocal'):
    """Calculates the transfer entropy between two variables from a set of
    vectors already calculated.

    """

    # First do an example for the case of k = l = 1
    # TODO: Sum loops to allow for a general case

    # Get the number of observations
    numobs = len(x_pred)

    if approach == 'averageoflocal':

        tesum = 0.0
        for x_pred_val, x_hist_val, y_hist_val in zip(x_pred,
                                                      x_hist,
                                                      y_hist):

            sumelement = te_shu_elementcalc(x_pred, x_hist,
                                            y_hist,
                                            x_pred_val, x_hist_val, y_hist_val)
            tesum += sumelement
        tentropy = tesum / numobs

    elif approach == 'allsums':

        tesum = 0
        counter = 0
        print ((counter / len(x_pred_norm[0])) * 100), '%'
        for x_pred_val in x_pred_norm[0]:
            counter += 1.0
            for x_hist_val in x_hist_norm[0]:
                for y_hist_val in y_hist_norm[0]:
                    sum_element = te_shu_elementcalc(x_pred_norm, x_hist_norm,
                                                     y_hist_norm,
                                                     x_pred_val, x_hist_val,
                                                     y_hist_val)
                    tesum = tesum + sum_element
            print ((counter / len(x_pred_norm[0])) * 100), '%'
        tentropy = tesum

    return tentropy
