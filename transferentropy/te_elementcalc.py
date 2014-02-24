"""
Created on Mon Feb 24 14:39:51 2014

@author: Simon Streicher
"""
import numpy as np


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
    sum_element = coeff * np.log(logterm_num / logterm_den)

    #print sum_element

    # TODO: Need to find a proper way to correct for cases when PDFs return
    # nan or inf values.
    # Most of the PDF issues are associated with the x_hist values being
    # very similar to the x_pred values.
    # Some very small negative values are sometimes returned.

    if (str(sum_element[0]) == 'nan' or sum_element < 0
            or str(sum_element[0]) == 'inf'):
        sum_element = 0

    return sum_element
