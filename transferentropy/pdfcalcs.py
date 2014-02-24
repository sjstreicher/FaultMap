# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:38:16 2014

@author: s13071832
"""
import numpy as np
from scipy import stats


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
