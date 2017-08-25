# -*- coding: utf-8 -*-
"""This module stores the weight calculator classes
used by the gaincalc module.

"""
# Standard libraries
import logging

import numpy as np

import data_processing
import transentropy


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
                            'signchange', 'threshcorr', 'threshdir',
                            'threshpass', 'directionpass', 'dirval']

    def calcweight(self, causevardata, affectedvardata, *args):
        """Calculates the correlation between two vectors containing
        timer series data.

        """

        corrval = np.corrcoef(causevardata.T, affectedvardata.T)[1, 0]
        return [corrval], None

    def calcsigthresh(self, *_):
        return [self.threshcorr]

    def report(self, weightcalcdata, causevarindex, affectedvarindex,
               weightlist, _):

        """Calculates and reports the relevant output for each combination
        of variables tested.

        """
        variables = weightcalcdata.variables
        causevar = variables[causevarindex]
        affectedvar = variables[affectedvarindex]

        if weightcalcdata.bidirectional_delays:
            baseval = weightlist[(len(weightlist) / 2)]
        else:
            baseval = weightlist[0]

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

        # Correlation thresholds from Bauer2008 Eq. 4
        maxcorr_abs = max(maxval, abs(minval))
        bestdelay = weightcalcdata.actual_delays[delay_index]
        if (maxval and minval) != 0:
            directionindex = 2 * (abs(maxval + minval) /
                                  (maxval + abs(minval)))
        else:
            directionindex = 0

        signchange = not ((baseval / weightlist[delay_index]) >= 0)

        logging.info("Maximum correlation value: " + str(maxval))
        logging.info("Minimum correlation value: " + str(minval))
        logging.info("The maximum correlation between " + causevar +
                     " and " + affectedvar + " is: " + str(maxcorr))
        logging.info("The corresponding delay is: " +
                     str(bestdelay))
        logging.info("The correlation with no delay is: " +
                     str(baseval))

        logging.info("Directionality value: " + str(directionindex))

        corrthreshpass = None
        dirthreshpass = None

        if weightcalcdata.sigtest:
            corrthreshpass = (maxcorr_abs >= self.threshcorr)
            dirthreshpass = ((directionindex >= self.threshdir)
                             and (bestdelay >= 0.))
            logging.info("Correlation threshold passed: " +
                         str(corrthreshpass))
            logging.info("Directionality threshold passed: " +
                         str(dirthreshpass))

        dataline = [causevar, affectedvar, baseval,
                    maxcorr, str(bestdelay), str(delay_index),
                    signchange, self.threshcorr, self.threshdir,
                    corrthreshpass, dirthreshpass, directionindex]

        return dataline


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
        in the sense that it requires the full dataset in order to execute.

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


class TransentWeightcalc(object):
    """This class provides methods for calculating the weights according to
    the transfer entropy method.

    """

    def __init__(self, weightcalcdata, estimator):
        self.data_header = ['causevar', 'affectedvar', 'base_ent',
                            'max_ent', 'max_delay', 'max_index', 'threshold',
                            'bias_mean', 'bias_std',
                            'threshpass', 'directionpass',
                            'k_hist_fwd', 'k_tau_fwd', 'l_hist_fwd',
                            'l_tau_fwd', 'delay_fwd',
                            'k_hist_bwd', 'k_tau_bwd', 'l_hist_bwd',
                            'l_tau_bwd', 'delay_bwd']

        self.estimator = estimator
        self.infodynamicsloc = weightcalcdata.infodynamicsloc
        if weightcalcdata.sigtest:
            self.te_thresh_method = weightcalcdata.te_thresh_method

        if self.estimator == 'kraskov':
            self.parameters = weightcalcdata.additional_parameters

        # Test if parameters dictionary exists
        try:
            self.parameters.keys()
        except:
            self.parameters = {}

        # Add kernel bandwidth to parameters
        if ((self.estimator == 'kernel') and
                (weightcalcdata.kernel_width is not None)):
            self.parameters['kernel_width'] = \
                weightcalcdata.kernel_width

    def calcweight(self, causevardata, affectedvardata, weightcalcdata,
                   causevarindex, affectedvarindex):
        """"Calculates the transfer entropy between two vectors containing
        timer series data.

        """
        # Calculate transfer entropy as the difference
        # between the forward and backwards entropy

        # Pass special estimator specific parameters in here

        transent_fwd, auxdata_fwd = \
            transentropy.calc_infodynamics_te(
                self.infodynamicsloc, self.estimator,
                affectedvardata.T, causevardata.T, **self.parameters)

        transent_bwd, auxdata_bwd = \
            transentropy.calc_infodynamics_te(
                self.infodynamicsloc, self.estimator,
                causevardata.T, affectedvardata.T, **self.parameters)

        transent_directional = transent_fwd - transent_bwd
        transent_absolute = transent_fwd

        return [transent_directional, transent_absolute], \
            [auxdata_fwd, auxdata_bwd]

    def select_weights(self, weightcalcdata, causevar, affectedvar,
                       weightlist, directional):

        if directional:
            directionstring = "directional"
        else:
            directionstring = "absolute"
        # Initiate flag indicating whether direction test passed
        directionpass = None

        if weightcalcdata.bidirectional_delays:
            baseval = weightlist[(len(weightlist) / 2)]

            # Get maximum weight in forward direction
            # This includes all positive delays including zero
            maxval_forward = \
                max(weightlist[(len(weightlist) - 1) / 2:])
            # Get maximum weight in backward direction
            # This includes all negative delays exluding zero
            maxval_backward = \
                max(weightlist[:(len(weightlist) - 1) / 2])

            delay_index_forward = weightlist.index(maxval_forward)
            delay_index_backward = weightlist.index(maxval_backward)

            bestdelay_forward = \
                weightcalcdata.actual_delays[delay_index_forward]
            bestdelay_backward = \
                weightcalcdata.actual_delays[delay_index_backward]

            # Test if the maximum forward value is bigger than the maximum
            # backward value
            if maxval_forward > maxval_backward:
                # We accept this value as true (pending significance testing)
                directionpass = True
            else:
                # Test whether the maximum forward delay is smaller than the
                # maximum backward delay
                # If this test passes we still consider the direction test to
                # pass
                if bestdelay_forward < abs(bestdelay_backward):
                    directionpass = True
                else:
                    directionpass = False

            # Assign forward values to generic outputs
            maxval = maxval_forward
            delay_index = delay_index_forward
            bestdelay = bestdelay_forward

            logging.info("The maximum forward " + directionstring +
                         " TE between " + causevar + " and " + affectedvar +
                         " is: " + str(maxval_forward))
            logging.info("The maximum backward " + directionstring +
                         " TE between " + causevar + " and " + affectedvar +
                         " is: " + str(maxval_backward))
            if directionpass is True:
                logging.info("The direction test passed")
            elif directionpass is False:
                logging.info("The direction test failed")

        else:
            baseval = weightlist[0]
            maxval = max(weightlist)
            delay_index = weightlist.index(maxval)
            bestdelay = weightcalcdata.actual_delays[delay_index]

            logging.info("The maximum " + directionstring + " TE between " +
                         causevar + " and " + affectedvar + " is: " +
                         str(maxval))

        bestdelay_sample = weightcalcdata.sample_delays[delay_index]

        return baseval, maxval, delay_index, bestdelay, \
            bestdelay_sample, directionpass

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

        # Get best weights and delays and an indication whether the
        # directionality test was passed for bidirectional testing cases

        # Do everything for the directional case
        baseval_directional, maxval_directional, delay_index_directional, \
            bestdelay_directional, bestdelay_sample_directional, \
            directionpass_directional = \
            self.select_weights(weightcalcdata, causevar,
                                affectedvar, weightlist_directional,
                                True)

        # Repeat for the absolute case
        baseval_absolute, maxval_absolute, delay_index_absolute, \
            bestdelay_absolute, bestdelay_sample_absolute, \
            directionpass_absolute = \
            self.select_weights(weightcalcdata, causevar,
                                affectedvar, weightlist_absolute,
                                False)

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
                threshent_directional, threshent_absolute = \
                    self.thresh_rankorder(
                        thresh_affectedvardata_directional.T,
                        thresh_causevardata.T)
            elif self.te_thresh_method == 'sixsigma':
                threshent_directional, threshent_absolute = \
                    self.thresh_sixsigma(
                        thresh_affectedvardata_directional.T,
                        thresh_causevardata.T)

            logging.info("The directional TE threshold is: " +
                         str(threshent_directional[0]))

            if maxval_directional >= threshent_directional[0] \
                    and maxval_directional > 0:
                threshpass_directional = True
            else:
                threshpass_directional = False

            if not delay_index_directional == delay_index_absolute:
                # Need to do own calculation of absolute significance
                if self.te_thresh_method == 'rankorder':
                    _, threshent_absolute = \
                        self.thresh_rankorder(
                            thresh_affectedvardata_absolute.T,
                            thresh_causevardata.T)
                elif self.te_thresh_method == 'sixsigma':
                    _, threshent_absolute = \
                        self.thresh_sixsigma(
                            thresh_affectedvardata_absolute.T,
                            thresh_causevardata.T)

            logging.info("The absolute TE threshold is: " +
                         str(threshent_absolute[0]))

            if maxval_absolute >= threshent_absolute[0] \
                    and maxval_absolute > 0:
                threshpass_absolute = True
            else:
                threshpass_absolute = False

        elif not weightcalcdata.sigtest:
            threshent_directional = None
            threshent_absolute = None
            threshpass_directional = None
            threshpass_absolute = None
            directionpass_directional = None
            directionpass_absolute = None

        dataline_directional = \
            [causevar, affectedvar, baseval_directional,
             maxval_directional, bestdelay_directional,
             delay_index_directional, threshent_directional[0],
             threshent_directional[1], threshent_directional[2],
             threshpass_directional, directionpass_directional]

        dataline_absolute = \
            [causevar, affectedvar, baseval_absolute,
             maxval_absolute, bestdelay_absolute,
             delay_index_absolute, threshent_absolute[0],
             threshent_absolute[1], threshent_absolute[2],
             threshpass_absolute, directionpass_absolute]

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
            surr_tsdata = \
                [data_processing.gen_iaaft_surrogates(
                    original_causal, 10)
                 for n in range(num)]

        elif self.te_surr_method == 'random_shuffle':
            surr_tsdata = \
                [data_processing.shuffle_data(causal_data) for n in range(num)]

        surr_te_fwd = []
        surr_te_bwd = []
        for n in range(num):

            surr_te_fwd.append(transentropy.calc_infodynamics_te(
                self.infodynamicsloc, self.estimator,
                affected_data, surr_tsdata[n][0, :],
                **self.parameters)[0])

            surr_te_bwd.append(transentropy.calc_infodynamics_te(
                self.infodynamicsloc, self.estimator,
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

        threshent_directional = max(surr_te_directional)
        nullbias_directional = np.mean(surr_te_directional)
        nullstd_directional = np.std(surr_te_directional)

        threshent_absolute = max(surr_te_absolute)
        nullbias_absolute = np.mean(surr_te_absolute)
        nullstd_absolute = np.std(surr_te_absolute)

        return [threshent_directional, nullbias_directional, nullstd_directional], \
               [threshent_absolute, nullbias_absolute, nullstd_absolute]

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

        threshent_directional = (6 * surr_te_directional_stdev) + \
            surr_te_directional_mean

        threshent_absolute = (6 * surr_te_absolute_stdev) + \
            surr_te_absolute_mean

        return [threshent_directional, surr_te_directional_mean, surr_te_directional_stdev], \
               [threshent_absolute, surr_te_absolute_mean, surr_te_absolute_stdev]

    def calcsigthresh(self, weightcalcdata, affected_data, causal_data):
        # print affected_data
        # print causal_data
        self.te_thresh_method = weightcalcdata.te_thresh_method
        self.te_surr_method = weightcalcdata.te_surr_method
        if self.te_thresh_method == 'rankorder':
            threshent_directional, threshent_absolute = \
                self.thresh_rankorder(affected_data, causal_data)
        elif self.te_thresh_method == 'sixsigma':
            threshent_directional, threshent_absolute = \
                self.thresh_sixsigma(affected_data, causal_data)
        return [threshent_directional, threshent_absolute]
