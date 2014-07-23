# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:48:59 2014

@author: Simon
"""

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
