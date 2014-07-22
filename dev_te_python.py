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

from datagen import autoreg_gen


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


def kde_sklearn(x, x_grid, bandwidth=0.3, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


def kde_sklearn_create(x, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn creation"""
    # TOOD: Implement automatic optimal bandwidth selection
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.05, 2.0, 100)},
                        cv=20)  # 20-fold cross-validation
    grid.fit(x)
    kde_skl = grid.best_estimator_
    print grid.best_params_
#    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
#    kde_skl.fit(x[:, np.newaxis])
    return kde_skl


def kde_sklearn_sample(kde_skl, x_grid):
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

# Get sample data
data = autoreg_gen(1000, 0)

data = data.T

[x_pred, x_hist, y_hist] = vectorselection(data, 0, 500)

# Normalise
x_pred = sklearn.preprocessing.scale(x_pred)
x_hist = sklearn.preprocessing.scale(x_hist)
y_hist = sklearn.preprocessing.scale(y_hist)

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
    #ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
    ax.hist(data[0, :], 30, fc='gray', histtype='bar', alpha=0.3, normed=True)
    #ax.legend(loc='upper left')

    # Plot probability distribution of destination (affected) signal
    fig, ax = plt.subplots()
    #ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
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
pdf_2_read = kde_sklearn_sample(pdf_2, reading_grid)
# Plot
fig, ax = plt.subplots()
ax.plot(x_grid, pdf_2_read, '.', linewidth=3, alpha=0.5,
        label='pdf_x_hist')
#ax.plot(y_grid, pdf_2_read, '.', linewidth=3, alpha=0.5,
#        label='pdf_y_hist')
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


## Estimate p(x_{i+h}, x_i, y_i)
#data_1 = np.vstack([x_pred, x_hist, y_hist])
#pdf_1 = kde_sklearn_create(data_1.T)
## Plot
#fig, ax = plt.subplots()
#ax.plot(y_hist_grid, pdf_y_hist_read, linewidth=3, alpha=0.5,
#        label='pdf_y_hist')
#ax.hist(data[0, :], 30, fc='gray', histtype='bar', alpha=0.3, normed=True)
#plt.show()






#from scipy.stats.distributions import norm
#
## The grid we'll use for plotting
#x_grid = np.linspace(-4.5, 3.5, 1000)
#
## Draw points from a bimodal distribution in 1D
#np.random.seed(0)
#x = np.concatenate([norm(-1, 1.).rvs(400),
#                    norm(1, 0.3).rvs(100)])
#pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) +
#            0.2 * norm(1, 0.3).pdf(x_grid))
#
## Plot the kernel density estimate
#fig, ax = plt.subplots(1, 1, sharey=True,
#                       figsize=(13, 3))
#fig.subplots_adjust(wspace=0)
#
#for i in range(1):
#    pdf = kde_sklearn(x, x_grid, bandwidth=0.2)
#    ax.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
#    ax.fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
#    ax.set_title('Scikit-learn')
#    ax.set_xlim(-4.5, 3.5)
#
#grid = GridSearchCV(KernelDensity(),
#                    {'bandwidth': np.linspace(0.1, 1.0, 30)},
#                    cv=20)  # 20-fold cross-validation
#grid.fit(x[:, None])
#print grid.best_params_
#
#kde = grid.best_estimator_
#pdf = np.exp(kde.score_samples(x_grid[:, None]))
#
#fig, ax = plt.subplots()
#ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
#ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
#ax.legend(loc='upper left')
#ax.set_xlim(-4.5, 3.5)


def pdfcalcs(x_pred, x_hist, y_hist):
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
    pdf_1 = kde_sklearn_create(data_1)

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
