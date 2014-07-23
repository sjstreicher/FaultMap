# -*- coding: utf-8 -*-
"""This method is an implementation of transfer entropy calculation making use
of Python libraries only.

The transfer entropy is calculated according to the global average of local
entropies method as presented in Lizier2008.

@author: Simon Streicher
"""

import numpy as np
import sklearn
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt

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


def kde_sklearn_create(x, bandwidth=0.2, autobandwidth=True, **kwargs):
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


def pdfcalcs(x_pred, x_hist, y_hist):
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


def te_calc(pdf_1, pdf_2, pdf_3, pdf_4):
    """Calculate elements for summation for a specific set of coordinates"""

    logterm_num = (pdf_1 / pdf_2)
    logterm_den = (pdf_3 / pdf_4)
#    np.seterr(divide='ignore', invalid='ignore')
    sum_elements = np.sum(np.log2(logterm_num / logterm_den))
#    np.seterr(divide=None, invalid=None)
    tentropy = (sum_elements / len(pdf_1))
    return tentropy


def calc_python_te(x_pred, x_hist, y_hist):
    [pdf_1, pdf_2, pdf_3, pdf_4] = pdfcalcs(x_pred, x_hist, y_hist)
    tentropy = te_calc(pdf_1, pdf_2, pdf_3, pdf_4)
    return tentropy

## Get sample data
#[x_pred, x_hist, y_hist] = autoreg_datagen(5, 0, 2500, 1000)
#
## Normalise
##x_pred = sklearn.preprocessing.scale(x_pred)
##x_hist = sklearn.preprocessing.scale(x_hist)
##y_hist = sklearn.preprocessing.scale(y_hist)
#
#x_pred_norm = sklearn.preprocessing.scale(x_pred, axis=1)
#x_hist_norm = sklearn.preprocessing.scale(x_hist, axis=1)
#y_hist_norm = sklearn.preprocessing.scale(y_hist, axis=1)
#
#result = calc_python_te(x_pred_norm[0], x_hist_norm[0], y_hist_norm[0])
#print result

plotting = False
if plotting:

    # Plot sample data and random walk
    fig, ax = plt.subplots()
    ax.plot(range(data.shape[1]), data[0, :], linewidth=3, alpha=0.5,
            label='source')
    ax.plot(range(data.shape[1]), data[1, :], linewidth=3, alpha=0.5,
            label='destination')
    ax.legend(loc='upper left')

    # Plot probability distribution of source (causal) signal
    fig, ax = plt.subplots()
    ax.hist(data[0, :], 30, fc='gray', histtype='bar', alpha=0.3, normed=True)
    #ax.legend(loc='upper left')

    # Plot probability distribution of destination (affected) signal
    fig, ax = plt.subplots()
    ax.hist(data[1, :], 30, fc='gray', histtype='bar', alpha=0.3, normed=True)
    #ax.legend(loc='upper left')

    # Plot x_pred and x_hist
    fig, ax = plt.subplots()
    ax.plot(range(len(x_pred)), x_pred, linewidth=3, alpha=0.5,
            label='x_pred')
    ax.plot(range(len(x_hist)), x_hist, linewidth=3, alpha=0.5,
            label='x_hist')
    ax.plot(range(len(y_hist)), y_hist, linewidth=3, alpha=0.5,
            label='y_hist')
    ax.legend(loc='upper left')

    #y_hist_grid = np.linspace(-4, 4, 1000)
    y_hist_grid = y_hist

    # Estimate density of source signal
    pdf_y_hist = kde_sklearn_create(y_hist[:, None])
    pdf_y_hist_read = kde_sklearn_sample(pdf_y_hist, y_hist_grid)
    # Plot together with histogram
    fig, ax = plt.subplots()
    ax.plot(y_hist_grid, pdf_y_hist_read, '.', linewidth=3, alpha=0.5,
            label='pdf_y_hist')
    ax.hist(data[0, :], 30, fc='gray', histtype='bar', alpha=0.3, normed=True)

    x_grid = np.arange(-4, 4, 0.1)
    y_grid = np.arange(-4, 4, 0.1)
    xx, yy = np.meshgrid(x_grid, y_grid)
    # Estimate p(x_i, y_i)
    data_2 = np.vstack([x_hist, y_hist])
    pdf_2 = kde_sklearn_create(data_2.T)
    reading_grid = np.vstack([x_grid, y_grid]).T
    pdf_2_read = kde_sklearn_sample(pdf_2, data_2.T)
    # Plot
    fig, ax = plt.subplots()
    ax.plot(x_hist, pdf_2_read, '.', linewidth=3, alpha=0.5,
            label='pdf_x_hist')
    ax.plot(y_hist, pdf_2_read, '.', linewidth=3, alpha=0.5,
            label='pdf_y_hist')
    plt.show()

    testarea = np.arange(-4, 4, 0.1)
    z = np.zeros([len(testarea), len(testarea)])
    for i in range(len(testarea)):
        for j in range(len(testarea)):
            z[i, j] = kde_sklearn_sample(pdf_2, np.array([[testarea[i],
                                                          testarea[j]]]))

    plt.figure()
    #z =  kde_sklearn_sample(pdf_2, reading_grid)
    CS = plt.contour(testarea, testarea, z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Simplest default with labels')
    plt.show()

    # Plot pdf2 over N
    fig, ax = plt.subplots()
    ax.plot(range(len(x_hist)), pdf_2_read, '.', linewidth=3, alpha=0.5,
            label='pdf_2')
    plt.show()

    # Estimate p(x_{i+h}, x_i, y_i)
    data_1 = np.vstack([x_pred, x_hist, y_hist])
    pdf_1 = kde_sklearn_create(data_1.T)
    pdf_1_read = kde_sklearn_sample(pdf_1, data_1.T)
    # Plot
    fig, ax = plt.subplots()
    ax.plot(range(len(x_hist)), pdf_1_read, '.', linewidth=3, alpha=0.5,
            label='pdf_1')
    plt.show()

    # Plot pdf_1 / pdf_2
    fig, ax = plt.subplots()
    ax.plot(range(len(x_hist)), pdf_1_read / pdf_2_read, '.', linewidth=3,
            alpha=0.5,
            label='log term numerator')
    plt.show()







#def calculate_te(delay, timelag, samples, sub_samples, mcsamples, k=1, l=1):
#    """Calculates the transfer entropy for a specific timelag (equal to
#    prediction horison) for a set of autoregressive data.
#
#    sub_samples is the amount of samples in the dataset used to calculate the
#    transfer entropy between two vectors (taken from the end of the dataset).
#    sub_samples <= samples
#
#    Currently only supports k = 1; l = 1;
#
#    You can search through a set of timelags in an attempt to identify the
#    original delay.
#    The transfer entropy should have a maximum value when timelag = delay
#    used to generate the autoregressive dataset.
#
#    """
#    # Get autoregressive datasets
#    data = getdata(samples, delay)
#
#    [x_pred, x_hist, y_hist] = vectorselection(data, timelag,
#                                               sub_samples, k, l)
#
#    transentropy = te_calc(x_pred, x_hist, y_hist, mcsamples)
#
#    return transentropy
