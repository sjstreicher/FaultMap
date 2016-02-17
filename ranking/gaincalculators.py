# -*- coding: utf-8 -*-
"""This module stores the weight calculator classes
used by the gaincalc module.

"""
# Standard libraries
import sys
import os
import numpy as np
import logging

# Own libraries
import transentropy

# Non-standard external libraries
from contextlib import contextmanager

# Own libraries
from data_processing import shuffle_data


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

with suppress_stdout():
    import pygeonetwork


class CorrWeightcalc(object):
    """This class provides methods for calculating the weights according to the
    cross-correlation method.

    """
    def __init__(self, weightcalcdata):
        """Read the files or functions and returns required data fields.

        """
        self.threshcorr = (1.85*(weightcalcdata.testsize**(-0.41))) + \
            (2.37*(weightcalcdata.testsize**(-0.53)))
        self.threshdir = 0.46*(weightcalcdata.testsize**(-0.16))
#        logging.info("Directionality threshold: " + str(self.threshdir))
#        logging.info("Correlation threshold: " + str(self.threshcorr))

        self.data_header = ['causevar', 'affectedvar', 'base_corr',
                            'max_corr', 'max_delay', 'max_index',
                            'signchange', 'corrthreshpass',
                            'dirrthreshpass', 'dirval']

    def calcweight(self, causevardata, affectedvardata, weightcalcdata,
                   causevarindex, affectedvarindex):
        """Calculates the correlation between two vectors containing
        timer series data.

        """

        corrval = np.corrcoef(causevardata.T, affectedvardata.T)[1, 0]
        return [corrval], None

    def calcsigthresh(self, _, affected_data, causal_data):
        return [self.threshcorr]

    def report(self, weightcalcdata, causevarindex, affectedvarindex,
               weightlist, proplist, datastore):

        """Calculates and reports the relevant output for each combination
        of variables tested.

        """
        variables = weightcalcdata.variables
        causevar = variables[causevarindex]
        affectedvar = variables[affectedvarindex]

        maxval = max(weightlist)
        minval = min(weightlist)
        # Value used to break tie between maxval and minval if 1 and -1
        tol = 0.
        # Always select maxval if both are equal
        # This is to ensure that 1 is selected above -1
        if (maxval + (minval + tol)) >= 0:
            maxcorr = maxval
        else:
            maxcorr = minval

        delay_index = weightlist.index(maxcorr)

        # Correlation thresholds from Bauer2008 eq. 4
        maxcorr_abs = max(maxval, abs(minval))
        bestdelay = weightcalcdata.actual_delays[delay_index]
        if not (maxval and minval) != 0:
            directionindex = 2 * (abs(maxval + minval) /
                                  (maxval + abs(minval)))
        else:
            directionindex = 0

        signchange = not ((weightlist[0] / weightlist[delay_index]) >= 0)
#        corrthreshpass = (maxcorr_abs >= self.threshcorr)
#        dirthreshpass = (directionindex >= self.threshdir)

        logging.info("Maximum correlation value: " + str(maxval))
        logging.info("Minimum correlation value: " + str(minval))
        logging.info("The maximum correlation between " + causevar +
                     " and " + affectedvar + " is: " + str(maxcorr))
        logging.info("The corresponding delay is: " +
                     str(bestdelay))
        logging.info("The correlation with no delay is: " +
                     str(weightlist[0]))

        logging.info("Directionality value: " + str(directionindex))

        corrthreshpass = None
        dirthreshpass = None
        if weightcalcdata.sigtest:
            corrthreshpass = (maxcorr_abs >= self.threshcorr)
            dirthreshpass = (directionindex >= self.threshdir)
            logging.info("Correlation threshold passed: " +
                         str(corrthreshpass))
            logging.info("Directionality threshold passed: " +
                         str(dirthreshpass))
            if not (corrthreshpass and dirthreshpass):
                maxcorr = 0

#        weight_array[affectedvarindex, causevarindex] = maxcorr

#        # Replace all nan by zero
#        nanlocs = np.isnan(weight_array)
#        weight_array[nanlocs] = 0

#        delay_array[affectedvarindex, causevarindex] = bestdelay

        dataline = [causevar, affectedvar, str(weightlist[0]),
                    maxcorr, str(bestdelay), str(delay_index),
                    signchange, corrthreshpass, dirthreshpass, directionindex]

#        datastore.append(dataline)

        return None


class PartialCorrWeightcalc(CorrWeightcalc):
    """This class provides methods for calculating the weights according to
    the partial correlation method.

    The option of handling delays in similar fashion to that of
    cross-correlation is provided. However, the ambiguity of using any delays
    in the partial correlation calculation should be noted.

    """

    def __init__(self, weightcalcdata):
        """Read the files or functions and returns required data fields.

        """
        super(PartialCorrWeightcalc, self).__init__(self, weightcalcdata)

        self.connections_used = weightcalcdata.connections_used

    def calcweight(self, causevardata, affectedvardata, weightcalcdata,
                   causevarindex, affectedvarindex):
        """Calculates the partial correlation between two elements in the
        complete dataset.

        It is important to note that this function differs from the others
        in the sense that it requires the full dataset to be known.

        """

        startindex = weightcalcdata.startindex
        size = weightcalcdata.testsize
        vardims = len(weightcalcdata.variables)

        # Get inputdata and initial connectionmatrix
        calcdata = (weightcalcdata.inputdata[:, :]
                    [startindex:startindex+size])

        newvariables = weightcalcdata.variables

        if self.connections_used:
            newconnectionmatrix = weightcalcdata.connectionmatrix
        else:
            # If there is not connection matrix given, create a fully
            # connected connection matrix
            newconnectionmatrix = np.ones((vardims, vardims))

        # Delete all columns not listed in causevarindexes
        # Calculation is cheap, and in order to keep things simple the
        # causevarindexes are the sole means of doing selective calculation
        dellist = []
        for index in range(vardims):
            if index not in weightcalcdata.causevarindexes:
                dellist.append(index)
                logging.info("Deleted column " + str(index))

        # Delete all columns listed in dellist from calcdata
        newcalcdata = np.delete(calcdata, dellist, 1)

        # Delete all indexes listed in dellist from variables
        newvariables = np.delete(newvariables, dellist)

        # Delete all rows and columns listed in dellist
        # from connectionmatrix
        newconnectionmatrix = np.delete(newconnectionmatrix, dellist, 1)
        newconnectionmatrix = np.delete(newconnectionmatrix, dellist, 0)

        # Calculate correlation matrix
        correlationmatrix = np.corrcoef(newcalcdata.T)
        # Calculate partial correlation matrix
        p_matrix = np.linalg.inv(correlationmatrix)
        d = p_matrix.diagonal()
        partialcorrelationmatrix = \
            np.where(newconnectionmatrix,
                     -p_matrix/np.abs(np.sqrt(np.outer(d, d))), 0)

        return partialcorrelationmatrix[affectedvarindex, causevarindex], None


class TransentWeightcalc:
    """This class provides methods for calculating the weights according to
    the transfer entropy method.

    """

    def __init__(self, weightcalcdata, estimator):
        self.data_header = ['causevar', 'affectedvar', 'base_ent',
                            'max_ent', 'max_delay', 'max_index', 'threshold',
                            'threshpass',
                            'k_hist_fwd', 'k_tau_fwd', 'l_hist_fwd',
                            'l_tau_fwd', 'delay_fwd',
                            'k_hist_bwd', 'k_tau_bwd', 'l_hist_bwd',
                            'l_tau_bwd', 'delay_bwd']

        self.estimator = estimator
        self.normalize = weightcalcdata.normalize
        self.infodynamicsloc = weightcalcdata.infodynamicsloc
        if weightcalcdata.sigtest:
            self.te_thresh_method = weightcalcdata.te_thresh_method

        if self.estimator == 'kraskov':
            self.parameters = weightcalcdata.additional_parameters
        else:
            self.parameters = {}

    def calcweight(self, causevardata, affectedvardata, weightcalcdata,
                   causevarindex, affectedvarindex):
        """"Calculates the transfer entropy between two vectors containing
        timer series data.

        """
        # Calculate transfer entropy as the difference
        # between the forward and backwards entropy

        # Pass special estimator specific parameters in here

        transent_fwd, auxdata_fwd = \
            transentropy.calc_infodynamics_te(self.infodynamicsloc,
                                              self.normalize,
                                              self.estimator,
                                              affectedvardata.T,
                                              causevardata.T,
                                              **self.parameters)

        transent_bwd, auxdata_bwd = \
            transentropy.calc_infodynamics_te(self.infodynamicsloc,
                                              self.normalize,
                                              self.estimator,
                                              causevardata.T,
                                              affectedvardata.T,
                                              **self.parameters)

        transent_directional = transent_fwd - transent_bwd
        transent_absolute = transent_fwd

        # Do not pass negatives on to weight array
        # TODO: Do this check later instead
#        if transent_directional < 0:
#            transent_directional = 0
#
#        if transent_absolute < 0:
#            transent_absolute = 0

        return [transent_directional, transent_absolute], \
            [auxdata_fwd, auxdata_bwd]

    def report(self, weightcalcdata, causevarindex, affectedvarindex,
               weightlist, proplist):

        """Calculates and reports the relevant output for each combination
        of variables tested.

        """

        variables = weightcalcdata.variables
        causevar = variables[causevarindex]
        affectedvar = variables[affectedvarindex]
        inputdata = weightcalcdata.inputdata

        # We already know that when dealing with transfer entropy
        # the weightlist will consist of a list of lists
        weightlist_directional, weightlist_absolute = weightlist

        proplist_fwd, proplist_bwd = proplist

        # Not supposed to differ among delay tests
        # TODO: Confirm this
#        k_hist_fwd, k_tau_fwd, l_hist_fwd, l_tau_fwd, delay_fwd = \
#            proplist_fwd[0]
#        k_hist_bwd, k_tau_bwd, l_hist_bwd, l_tau_bwd, delay_bwd = \
#            proplist_bwd[0]

        size = weightcalcdata.testsize
        startindex = weightcalcdata.startindex

        # Need placeholder in case significance is not tested
        threshpass_directional = None
        threshpass_absolute = None
        self.threshent_directional = None
        self.threshent_absolute = None

        # Do everything for the directional case
#            delay_array_directional = delay_array
        maxval_directional = max(weightlist_directional)
        delay_index_directional = \
            weightlist_directional.index(maxval_directional)
        bestdelay_directional = \
            weightcalcdata.actual_delays[delay_index_directional]
        bestdelay_sample_directional = \
            weightcalcdata.sample_delays[delay_index_directional]
#        delay_array_directional[affectedvarindex, causevarindex] = \
#            bestdelay_directional
        logging.info("The maximum directional TE between " + causevar +
                     " and " + affectedvar + " is: " +
                     str(maxval_directional))

        # Repeat for absolute case
#        delay_array_absolute = delay_array
        maxval_absolute = max(weightlist_absolute)
        delay_index_absolute = \
            weightlist_absolute.index(maxval_absolute)
        bestdelay_absolute = \
            weightcalcdata.actual_delays[delay_index_absolute]
        bestdelay_sample_absolute = \
            weightcalcdata.sample_delays[delay_index_absolute]
#        delay_array_absolute[affectedvarindex, causevarindex] = \
#            bestdelay_absolute
        logging.info("The maximum absolute TE between " + causevar +
                     " and " + affectedvar + " is: " +
                     str(maxval_absolute))

        if weightcalcdata.sigtest:
            self.te_thresh_method = weightcalcdata.te_thresh_method
            self.te_surr_method = weightcalcdata.te_surr_method
            # Calculate threshold for transfer entropy
            thresh_causevardata = \
                inputdata[:, causevarindex][startindex:startindex+size]
            thresh_affectedvardata_directional = \
                inputdata[:, affectedvarindex][startindex +
                                               bestdelay_sample_directional:
                                               startindex + size +
                                               bestdelay_sample_directional]

            thresh_affectedvardata_absolute = \
                inputdata[:, affectedvarindex][startindex +
                                               bestdelay_sample_absolute:
                                               startindex + size +
                                               bestdelay_sample_absolute]

            # Do significance calculations for directional case
            if self.te_thresh_method == 'rankorder':
                self.thresh_rankorder(thresh_affectedvardata_directional.T,
                                      thresh_causevardata.T)
            elif self.te_thresh_method == 'sixsigma':
                self.thresh_sixsigma(thresh_affectedvardata_directional.T,
                                     thresh_causevardata.T)

            logging.info("The directional TE threshold is: " +
                         str(self.threshent_directional))

            if maxval_directional >= self.threshent_directional \
                    and maxval_directional >= 0:
                threshpass_directional = True
            else:
                threshpass_directional = False
                maxval_directional = 0

            if not delay_index_directional == delay_index_absolute:
                # Need to do own calculation of absolute significance
                if self.te_thresh_method == 'rankorder':
                    self.thresh_rankorder(thresh_affectedvardata_absolute.T,
                                          thresh_causevardata.T)
                elif self.te_thresh_method == 'sixsigma':
                    self.thresh_sixsigma(thresh_affectedvardata_absolute.T,
                                         thresh_causevardata.T)

            logging.info("The absolute TE threshold is: " +
                         str(self.threshent_absolute))

            if maxval_absolute >= self.threshent_absolute \
                    and maxval_absolute >= 0:
                threshpass_absolute = True
            else:
                threshpass_absolute = False
                maxval_absolute = 0

#        weight_array[affectedvarindex, causevarindex] = maxval_directional

        dataline_directional = \
            [causevar, affectedvar, str(weightlist_directional[0]),
             maxval_directional, str(bestdelay_directional),
             str(delay_index_directional), self.threshent_directional,
             threshpass_directional]

        dataline_absolute = \
            [causevar, affectedvar, str(weightlist_absolute[0]),
             maxval_absolute, str(bestdelay_absolute),
             str(delay_index_absolute), self.threshent_absolute,
             threshpass_absolute]

        dataline_directional = dataline_directional + \
            proplist_fwd[delay_index_directional] + \
            proplist_bwd[delay_index_directional]

        dataline_absolute = dataline_absolute + \
            proplist_fwd[delay_index_absolute] + \
            proplist_bwd[delay_index_absolute]

        datalines = [dataline_directional, dataline_absolute]

        logging.info("The corresponding delay is: " +
                     str(bestdelay_directional))
        logging.info("The TE with no delay is: " + str(weightlist[0][0]))

        return datalines

    def calc_surr_te(self, affected_data, causal_data, num):
        """Calculates surrogate transfer entropy values for significance
        threshold purposes.

        Two methods for generating surrogate data is available:
        iAAFT (Schreiber 2000a) or random_shuffle in time.

        Returns list of surrogate transfer entropy values of length num.

        """

        # The causal (or source) data is replaced by surrogate data,
        # while the affected (or destination) data remains unchanged.

        # Get the causal data in the correct format
        # for surrogate generation
        original_causal = np.zeros((1, len(causal_data)))
        original_causal[0, :] = causal_data

        if self.te_surr_method == 'iAAFT':
            # Create surrogate data generation object
            iaaft_surrogate_gen = \
                pygeonetwork.surrogates.Surrogates(original_causal,
                                                   silence_level=2)

            surr_tsdata = \
                [iaaft_surrogate_gen.get_refined_AAFT_surrogates(
                    original_causal, 10)
                 for n in range(num)]

        elif self.te_surr_method == 'random_shuffle':
            surr_tsdata = \
                [shuffle_data(causal_data) for n in range(num)]

        surr_te_fwd = []
        surr_te_bwd = []
        for n in range(num):

            surr_te_fwd.append(transentropy.calc_infodynamics_te(
                    self.infodynamicsloc, self.normalize, self.estimator,
                    affected_data, surr_tsdata[n][0, :], **self.parameters)[0])

            surr_te_bwd.append(transentropy.calc_infodynamics_te(
                    self.infodynamicsloc, self.normalize, self.estimator,
                    surr_tsdata[n][0, :], affected_data,
                    **self.parameters)[0])

        surr_te_directional = \
            [surr_te_fwd[n] - surr_te_bwd[n] for n in range(num)]

        surr_te_absolute = [surr_te_fwd[n] for n in range(num)]

        return surr_te_directional, surr_te_absolute

    def thresh_rankorder(self, affected_data, causal_data):
        """Calculates the minimum threshold required for a transfer entropy
        value to be considered significant.

        Makes use of a 95% single-sided certainty and a rank-order method.
        This correlates to taking the maximum transfer entropy from 19
        surrogate transfer entropy calculations as the threshold,
        see Schreiber2000a.

        Alternatively, the second highest from 38 observations can be taken,
        etc.

        """
        surr_te_directional, surr_te_absolute = \
            self.calc_surr_te(affected_data, causal_data, 19)

        self.threshent_directional = max(surr_te_directional)
        self.threshent_absolute = max(surr_te_absolute)

    def thresh_sixsigma(self, affected_data, causal_data):
        """Calculates the minimum threshold required for a transfer entropy
        value to be considered significant.

        Makes use of a six sigma Gaussian check as done in Bauer2005 with 30
        samples of surrogate data.

        """
        surr_te_directional, surr_te_absolute = \
            self.calc_surr_te(affected_data, causal_data, 30)

        surr_te_directional_mean = np.mean(surr_te_directional)
        surr_te_directional_stdev = np.std(surr_te_directional)

        surr_te_absolute_mean = np.mean(surr_te_absolute)
        surr_te_absolute_stdev = np.std(surr_te_absolute)

        self.threshent_directional = (6 * surr_te_directional_stdev) + \
            surr_te_directional_mean

        self.threshent_absolute = (6 * surr_te_absolute_stdev) + \
            surr_te_absolute_mean

    def calcsigthresh(self, weightcalcdata, affected_data, causal_data):
        self.te_surr_method = weightcalcdata.te_surr_method
        self.thresh_rankorder(affected_data, causal_data)
        return [self.threshent_directional, self.threshent_absolute]
