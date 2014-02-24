"""
Created on Mon Feb 24 14:42:02 2014

@author: Simon Streicher
"""
import pdfcalcs
import te_elementcalc
import mcint
import random


def te_calc(x_pred, x_hist, y_hist, mcsamples):
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

    # Do triple integration using scipy.integrate.tplquad
    # Also do a version using mcint
    # See this for higher orders:
    # http://stackoverflow.com/questions/14071704/integrating-a-multidimensional-integral-in-scipy

    for nmc in [mcsamples]:
        random.seed(1)
        result, error = mcint.integrate(integrand, sampler(),
                                        measure=domainsize, n=nmc)

        print "Using n = ", nmc
        print "Result = ", result
    return result
