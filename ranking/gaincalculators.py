# -*- coding: utf-8 -*-
"""This module stores the weight calculator classes
used by the gaincalc module.

"""
# Standard libraries
import logging

import numpy as np

import transentropy
from ranking import data_processing


class CorrWeightcalc(object):
    """This class provides methods for calculating the weights according to the
    cross-correlation method.

    Calculates correlation using covariance with optional standardisation
    and detrending. Allows for effect of Skogestad scaling to be reflected
    in final result.

    """

    def __init__(self, weightcalcdata):
        """Read the files or functions and returns required data fields.

        """
        # These are the Bauer2005 thresholds
        # Due to the decision to make use of non-standardised correlation, a normal surrogate
        # thresholding approach will be used for each individual pair
        # self.threshcorr = (1.85*(weightcalcdata.testsize**(-0.41))) + \
        #     (2.37*(weightcalcdata.testsize**(-0.53)))
        # self.threshdir = 0.46*(weightcalcdata.testsize**(-0.16))
        #        logging.info("Directionality threshold: " + str(self.threshdir))
        #        logging.info("Correlation threshold: " + str(self.threshcorr))

        self.data_header = [
            "causevar",
            "affectedvar",
            "base_corr",
            "max_corr",
            "max_delay",
            "max_index",
            "signchange",
            "threshcorr",
            "threshdir",
            "threshpass",
            "directionpass",
            "dirval",
        ]

        if weightcalcdata.sigtest:
            self.thresh_method = weightcalcdata.thresh_method
            self.surr_method = weightcalcdata.surr_method

    def calcweight(self, causevardata, affectedvardata, *_):
        """Calculates the correlation between two vectors containing
        timer series data.

        """

        # corrval = np.corrcoef(causevardata.T, affectedvardata.T)[1, 0]
        # TODO: Provide the option of scaling the correlation measure
        # Un-normalised measure
        corrval = np.cov(causevardata.T, affectedvardata.T)[1, 0]
        # Normalised measure
        # corrval = np.corrcoef(causevardata.T, affectedvardata.T)[1, 0]
        # Here we use the biased correlation measure
        # corrval = np.correlate(causevardata.T, affectedvardata.T)[0] / len(affectedvardata)

        return [corrval], None

    # def calcsigthresh(self, *_):
    #     return [self.threshcorr]

    def calc_surr_correlation(self, weightcalcdata, causevar, affectedvar, box, trials):
        """Calculates surrogate correlation values for significance
        threshold purposes.

        Two methods for generating surrogate data is available:
        iAAFT (Schreiber 2000a) or random_shuffle in time.

        Returns list of surrogate correlation entropy values of length num.

        """

        # The causal (or source) data is replaced by surrogate data,
        # while the affected (or destination) data remains unchanged.

        # Generate surrogate causal data
        thresh_causevardata = box[:, weightcalcdata.variables.index(causevar)][
            weightcalcdata.startindex : weightcalcdata.startindex
            + weightcalcdata.testsize
        ]

        # Get the causal data in the correct format
        # for surrogate generation
        original_causal = np.zeros((1, len(thresh_causevardata)))
        original_causal[0, :] = thresh_causevardata

        if self.surr_method == "iAAFT":
            surr_tsdata = [
                data_processing.gen_iaaft_surrogates(original_causal, 10)
                for n in range(trials)
            ]
        elif self.surr_method == "random_shuffle":
            surr_tsdata = [
                data_processing.shuffle_data(thresh_causevardata) for n in range(trials)
            ]

        surr_corr_list = []
        surr_dirindex_list = []
        for n in range(trials):
            # Compute the weightlist for every trial by evaluating all delays
            surr_weightlist = []
            for delay_index in weightcalcdata.sample_delays:
                thresh_affectedvardata = box[
                    :, weightcalcdata.variables.index(affectedvar)
                ][
                    weightcalcdata.startindex
                    + delay_index : weightcalcdata.startindex
                    + weightcalcdata.testsize
                    + delay_index
                ]
                surr_weightlist.append(
                    self.calcweight(surr_tsdata[n][0, :], thresh_affectedvardata)[0][0]
                )

            _, maxcorr, _, _, _, directionindex, _ = self.select_weights(
                weightcalcdata, causevar, affectedvar, surr_weightlist
            )

            surr_corr_list.append(abs(maxcorr))
            surr_dirindex_list.append(directionindex)

        return surr_corr_list, surr_dirindex_list

    def thresh_rankorder(self, surr_corr, surr_dirindex):
        """Calculates the minimum threshold required for a correlation
        value to be considered significant.

        Makes use of a 95% single-sided certainty and a rank-order method.
        This correlates to taking the maximum transfer entropy from 19
        surrogate transfer entropy calculations as the threshold,
        see Schreiber2000a.

        Alternatively, the second highest from 38 observations can be taken,
        etc.

        """

        thresh_corr = np.percentile(surr_corr, 95)
        nullbias_corr = np.mean(surr_corr)
        nullstd_corr = np.std(surr_corr)

        # We can lower this limit to the 70th percentile to be more in line
        # with the idea of one standard deviation
        thresh_dirindex = np.percentile(surr_dirindex, 70)
        nullbias_dirindex = np.mean(surr_dirindex)
        nullstd_dirindex = np.std(surr_dirindex)

        return (
            [thresh_corr, nullbias_corr, nullstd_corr],
            [thresh_dirindex, nullbias_dirindex, nullstd_dirindex],
        )

    def thresh_stdevs(self, surr_corr, surr_dirindex, stdevs):
        """Calculates the minimum threshold required for a transfer entropy
        value to be considered significant.

        Makes use of a six sigma Gaussian check as done in Bauer2005 with 30
        samples of surrogate data.

        """

        surr_corr_mean = np.mean(surr_corr)
        surr_corr_stdev = np.std(surr_corr)

        surr_dirindex_mean = np.mean(surr_dirindex)
        surr_dirindex_stdev = np.std(surr_dirindex)

        thresh_corr = (stdevs * surr_corr_stdev) + surr_corr_mean

        # Their later work suggests only one standard deviation from mean
        thresh_dirindex = (1 * surr_dirindex_stdev) + surr_dirindex_mean

        return (
            [thresh_corr, surr_corr_mean, surr_corr_stdev],
            [thresh_dirindex, surr_dirindex_mean, surr_dirindex_stdev],
        )

    def calcsigthresh(self, weightcalcdata, causevar, affectedvar, box, _):

        # Calculates correlation as well as correlation directionality index
        # significance thresholds
        # The correlation threshold is a simple statistical significance test
        # based on surrogate results
        # The directionality threshold calculation is as described by Bauer2005

        # This is only called when the entropy at each delay is tested

        if self.thresh_method == "rankorder":
            surr_corr, surr_dirindex = self.calc_surr_correlation(
                weightcalcdata, causevar, affectedvar, box, 19
            )
            thresh_corr, thresh_dirindex = self.thresh_rankorder(
                surr_corr, surr_dirindex
            )
        elif self.thresh_method == "stdevs":
            surr_corr, surr_dirindex = self.calc_surr_correlation(
                weightcalcdata, causevar, affectedvar, box, 30
            )
            thresh_corr, thresh_dirindex = self.thresh_stdevs(
                surr_corr, surr_dirindex, 3
            )
        else:
            raise ValueError("Threshold method not recognized")

        return [thresh_corr[0], thresh_dirindex[0]]

    @staticmethod
    def select_weights(weightcalcdata, causevar, affectedvar, weightlist):

        # Initiate flag indicating whether direction test passed
        directionpass = None

        if weightcalcdata.bidirectional_delays:
            baseval = weightlist[int((len(weightlist) / 2))]
        else:
            baseval = weightlist[0]

        # Alternative interpretation:
        # Maxval is defined as the maximum absolute value in the forward direction
        # Minval is defined as the maximum absolute value in the negative direction
        if weightcalcdata.bidirectional_delays:
            maxval = max(np.abs(weightlist[int((len(weightlist) / 2)) :]))
            minval = -1 * max(np.abs(weightlist[: int((len(weightlist) / 2))]))
        else:
            raise ValueError(
                "The correlation directionality test is only defined for bidirectional delays"
            )

        # Test that maxval is positive and minval is negative
        if maxval < 0 or minval > 0:
            raise ValueError("Values do not adhere to sign expectations")
        # Value used to break tie between maxval and minval if 1 and -1
        tol = 0.0
        # Always select maxval if both are equal
        # This is to ensure that 1 is selected above -1
        if (maxval + (minval + tol)) >= 0:
            maxcorr = maxval
        else:
            maxcorr = abs(minval)

        delay_index = list(np.abs(weightlist)).index(maxcorr)

        # Correlation thresholds from Bauer2008 Eq. 4
        # maxcorr_abs = abs(maxcorr)
        bestdelay = weightcalcdata.actual_delays[delay_index]
        if (maxval and minval) != 0:
            directionindex = abs(maxval + minval) / ((maxval + abs(minval)) * 0.5)
        else:
            directionindex = 0

        signchange = not ((baseval / weightlist[delay_index]) >= 0)

        logging.info("Forwards maximum correlation value: " + str(maxval))
        logging.info("Backwards maximum correlation value: " + str(minval))
        logging.info(
            "The maximum correlation between "
            + causevar
            + " and "
            + affectedvar
            + " is: "
            + str(maxcorr)
        )
        logging.info("The corresponding delay is: " + str(bestdelay))
        logging.info("The correlation with no delay is: " + str(baseval))

        logging.info("Directionality value: " + str(directionindex))

        bestdelay_sample = weightcalcdata.sample_delays[delay_index]

        return (
            baseval,
            maxcorr,
            delay_index,
            bestdelay,
            bestdelay_sample,
            directionindex,
            signchange,
        )

    def report(
        self, weightcalcdata, causevarindex, affectedvarindex, weightlist, box, _
    ):

        """Calculates and reports the relevant output for each combination
        of variables tested.

        """
        causevar = weightcalcdata.variables[causevarindex]
        affectedvar = weightcalcdata.variables[affectedvarindex]

        # Get best weights and delays and an indication whether the
        # directionality test was passed

        baseval, maxcorr, delay_index, bestdelay, bestdelay_sample, directionindex, signchange = self.select_weights(
            weightcalcdata, causevar, affectedvar, weightlist
        )

        corrthreshpass = None
        dirthreshpass = None

        if weightcalcdata.sigtest:
            # Calculate value and directionality thresholds for correlation

            # In the case of correlation, we need to create one surrogate weightlist
            # (evaluation at all delays) for every trial, and get the maxcorr and
            # directionality index from it.

            # Do significance calculations

            if self.thresh_method == "rankorder":
                surr_corr, surr_dirindex = self.calc_surr_correlation(
                    weightcalcdata, causevar, affectedvar, box, 19
                )
                threshcorr, threshdir = self.thresh_rankorder(surr_corr, surr_dirindex)
            elif self.thresh_method == "stdevs":
                surr_corr, surr_dirindex = self.calc_surr_correlation(
                    weightcalcdata, causevar, affectedvar, box, 30
                )
                threshcorr, threshdir = self.thresh_stdevs(surr_corr, surr_dirindex, 3)

            logging.info("The correlation threshold is: " + str(threshcorr[0]))
            logging.info("The direction index threshold is: " + str(threshdir[0]))

            corrthreshpass = abs(maxcorr) >= threshcorr[0]
            dirthreshpass = (directionindex >= threshdir[0]) and (bestdelay >= 0.0)
            logging.info("Correlation threshold passed: " + str(corrthreshpass))
            logging.info("Directionality threshold passed: " + str(dirthreshpass))

        elif not weightcalcdata.sigtest:
            threshcorr = [None]
            threshdir = [None]

        # The maxcorr value can be positive or negative, the eigenvector ranking steps
        # needs to abs all the weights

        dataline = [
            causevar,
            affectedvar,
            baseval,
            maxcorr,
            str(bestdelay),
            str(delay_index),
            signchange,
            threshcorr[0],
            threshdir[0],
            corrthreshpass,
            dirthreshpass,
            directionindex,
        ]

        return dataline


# class PartialCorrWeightcalc(CorrWeightcalc):
#     """This class provides methods for calculating the weights according to
#     the partial correlation method.
#
#     The option of handling delays in similar fashion to that of
#     cross-correlation is provided. However, the ambiguity of using any delays
#     in the partial correlation calculation should be noted.
#
#     """
#
#     def __init__(self, weightcalcdata):
#         """Read the files or functions and returns required data fields.
#
#         """
#         super(PartialCorrWeightcalc, self).__init__(self, weightcalcdata)
#
#         self.connections_used = weightcalcdata.connections_used
#
#     def calcweight(self, causevardata, affectedvardata, weightcalcdata,
#                    causevarindex, affectedvarindex):
#         """Calculates the partial correlation between two elements in the
#         complete dataset.
#
#         It is important to note that this function differs from the others
#         in the sense that it requires the full dataset in order to execute.
#
#         """
#
#         startindex = weightcalcdata.startindex
#         size = weightcalcdata.testsize
#         vardims = len(weightcalcdata.variables)
#
#         # Get inputdata and initial connectionmatrix
#         calcdata = (weightcalcdata.inputdata[:, :]
#                     [startindex:startindex+size])
#
#         newvariables = weightcalcdata.variables
#
#         if self.connections_used:
#             newconnectionmatrix = weightcalcdata.connectionmatrix
#         else:
#             # If there is not connection matrix given, create a fully
#             # connected connection matrix
#             newconnectionmatrix = np.ones((vardims, vardims))
#
#         # Delete all columns not listed in causevarindexes
#         # Calculation is cheap, and in order to keep things simple the
#         # causevarindexes are the sole means of doing selective calculation
#         dellist = []
#         for index in range(vardims):
#             if index not in weightcalcdata.causevarindexes:
#                 dellist.append(index)
#                 logging.info("Deleted column " + str(index))
#
#         # Delete all columns listed in dellist from calcdata
#         newcalcdata = np.delete(calcdata, dellist, 1)
#
#         # Delete all indexes listed in dellist from variables
#         newvariables = np.delete(newvariables, dellist)
#
#         # Delete all rows and columns listed in dellist
#         # from connectionmatrix
#         newconnectionmatrix = np.delete(newconnectionmatrix, dellist, 1)
#         newconnectionmatrix = np.delete(newconnectionmatrix, dellist, 0)
#
#         # Calculate correlation matrix
#         correlationmatrix = np.corrcoef(newcalcdata.T)
#         # Calculate partial correlation matrix
#         p_matrix = np.linalg.inv(correlationmatrix)
#         d = p_matrix.diagonal()
#         partialcorrelationmatrix = \
#             np.where(newconnectionmatrix,
#                      -p_matrix/np.abs(np.sqrt(np.outer(d, d))), 0)
#
#         return partialcorrelationmatrix[affectedvarindex, causevarindex], None


class TransentWeightcalc(object):
    """This class provides methods for calculating the weights according to
    the transfer entropy method.

    """

    def __init__(self, weightcalcdata, estimator):
        self.data_header = [
            "causevar",
            "affectedvar",
            "base_ent",
            "max_ent",
            "max_delay",
            "max_index",
            "threshold",
            "bias_mean",
            "bias_std",
            "threshpass",
            "directionpass",
            "k_hist_fwd",
            "k_tau_fwd",
            "l_hist_fwd",
            "l_tau_fwd",
            "delay_fwd",
            "k_hist_bwd",
            "k_tau_bwd",
            "l_hist_bwd",
            "l_tau_bwd",
            "delay_bwd",
            "mi_fwd",
            "mi_bwd",
        ]

        self.estimator = estimator
        self.infodynamicsloc = weightcalcdata.infodynamicsloc
        if weightcalcdata.sigtest:
            self.thresh_method = weightcalcdata.thresh_method
            self.surr_method = weightcalcdata.surr_method

        if self.estimator == "kraskov":
            parameters_dict = weightcalcdata.additional_parameters
            parameters_dict["use_gpu"] = weightcalcdata.use_gpu
            self.parameters = weightcalcdata.additional_parameters

        # Test if parameters dictionary exists
        try:
            self.parameters.keys()
        except:
            self.parameters = {}

        # Add kernel bandwidth to parameters
        if (self.estimator == "kernel") and (weightcalcdata.kernel_width is not None):
            self.parameters["kernel_width"] = weightcalcdata.kernel_width

    def calcweight(self, causevardata, affectedvardata, *_):
        """"Calculates the transfer entropy between two vectors containing
        timer series data.

        """
        # Calculate transfer entropy as the difference
        # between the forward and backwards entropy

        # Pass special estimator specific parameters in here

        transent_fwd, auxdata_fwd = transentropy.calc_infodynamics_te(
            self.infodynamicsloc,
            self.estimator,
            affectedvardata.T,
            causevardata.T,
            **self.parameters
        )

        transent_bwd, auxdata_bwd = transentropy.calc_infodynamics_te(
            self.infodynamicsloc,
            self.estimator,
            causevardata.T,
            affectedvardata.T,
            **self.parameters
        )

        transent_directional = transent_fwd - transent_bwd
        transent_absolute = transent_fwd

        return [transent_directional, transent_absolute], [auxdata_fwd, auxdata_bwd]

    @staticmethod
    def select_weights(weightcalcdata, causevar, affectedvar, weightlist, directional):

        if directional:
            directionstring = "directional"
        else:
            directionstring = "absolute"
        # Initiate flag indicating whether direction test passed
        directionpass = None

        if weightcalcdata.bidirectional_delays:
            baseval = weightlist[int(len(weightlist) / 2)]

            # Get maximum weight in forward direction
            # This includes all positive delays including zero
            maxval_forward = max(weightlist[int((len(weightlist) - 1) / 2) :])
            # Get maximum weight in backward direction
            # This includes all negative delays excluding zero
            maxval_backward = max(weightlist[: int((len(weightlist) - 1) / 2)])

            delay_index_forward = weightlist.index(maxval_forward)
            delay_index_backward = weightlist.index(maxval_backward)

            bestdelay_forward = weightcalcdata.actual_delays[delay_index_forward]
            bestdelay_backward = weightcalcdata.actual_delays[delay_index_backward]

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

            logging.info(
                "The maximum forward "
                + directionstring
                + " TE between "
                + causevar
                + " and "
                + affectedvar
                + " is: "
                + str(maxval_forward)
            )
            logging.info(
                "The maximum backward "
                + directionstring
                + " TE between "
                + causevar
                + " and "
                + affectedvar
                + " is: "
                + str(maxval_backward)
            )
            if directionpass is True:
                logging.info("The direction test passed")
            elif directionpass is False:
                logging.info("The direction test failed")

        else:
            baseval = weightlist[0]
            maxval = max(weightlist)
            delay_index = weightlist.index(maxval)
            bestdelay = weightcalcdata.actual_delays[delay_index]

            logging.info(
                "The maximum "
                + directionstring
                + " TE between "
                + causevar
                + " and "
                + affectedvar
                + " is: "
                + str(maxval)
            )

        bestdelay_sample = weightcalcdata.sample_delays[delay_index]

        return baseval, maxval, delay_index, bestdelay, bestdelay_sample, directionpass

    def report(
        self,
        weightcalcdata,
        causevarindex,
        affectedvarindex,
        weightlist,
        box,
        proplist,
        milist,
    ):

        """Calculates and reports the relevant output for each combination
        of variables tested.

        """

        variables = weightcalcdata.variables
        causevar = variables[causevarindex]
        affectedvar = variables[affectedvarindex]
        # inputdata = weightcalcdata.inputdata

        # We already know that when dealing with transfer entropy
        # the weightlist will consist of a list of lists
        weightlist_directional, weightlist_absolute = weightlist

        proplist_fwd, proplist_bwd = proplist
        milist_fwd, milist_bwd = milist

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
        baseval_directional, maxval_directional, delay_index_directional, bestdelay_directional, bestdelay_sample_directional, directionpass_directional = self.select_weights(
            weightcalcdata, causevar, affectedvar, weightlist_directional, True
        )

        # Repeat for the absolute case
        baseval_absolute, maxval_absolute, delay_index_absolute, bestdelay_absolute, bestdelay_sample_absolute, directionpass_absolute = self.select_weights(
            weightcalcdata, causevar, affectedvar, weightlist_absolute, False
        )

        if weightcalcdata.sigtest:
            # Calculate threshold for transfer entropy
            thresh_causevardata = box[:, causevarindex][startindex : startindex + size]
            thresh_affectedvardata_directional = box[:, affectedvarindex][
                startindex
                + bestdelay_sample_directional : startindex
                + size
                + bestdelay_sample_directional
            ]

            thresh_affectedvardata_absolute = box[:, affectedvarindex][
                startindex
                + bestdelay_sample_absolute : startindex
                + size
                + bestdelay_sample_absolute
            ]

            # Do significance calculations for directional case
            if self.thresh_method == "rankorder":
                surr_te_directional, surr_te_absolute = self.calc_surr_te(
                    weightcalcdata,
                    causevar,
                    affectedvar,
                    box,
                    bestdelay_sample_directional,
                    19,
                )
                threshent_directional, threshent_absolute = self.thresh_rankorder(
                    surr_te_directional, surr_te_absolute
                )
            elif self.thresh_method == "stdevs":
                surr_te_directional, surr_te_absolute = self.calc_surr_te(
                    weightcalcdata,
                    causevar,
                    affectedvar,
                    box,
                    bestdelay_sample_directional,
                    30,
                )
                threshent_directional, threshent_absolute = self.thresh_stdevs(
                    surr_te_directional, surr_te_absolute, 3
                )

            logging.info(
                "The directional TE threshold is: " + str(threshent_directional[0])
            )

            if (
                maxval_directional >= threshent_directional[0]
                and maxval_directional > 0
            ):
                threshpass_directional = True
            else:
                threshpass_directional = False

            if not delay_index_directional == delay_index_absolute:
                # Need to do own calculation of absolute significance
                if self.thresh_method == "rankorder":
                    surr_te_directional, surr_te_absolute = self.calc_surr_te(
                        weightcalcdata,
                        causevar,
                        affectedvar,
                        box,
                        bestdelay_sample_absolute,
                        19,
                    )
                    _, threshent_absolute = self.thresh_rankorder(
                        surr_te_directional, surr_te_absolute
                    )
                elif self.thresh_method == "stdevs":
                    surr_te_directional, surr_te_absolute = self.calc_surr_te(
                        weightcalcdata,
                        causevar,
                        affectedvar,
                        box,
                        bestdelay_sample_absolute,
                        30,
                    )
                    _, threshent_absolute = self.thresh_stdevs(
                        surr_te_directional, surr_te_absolute, 3
                    )

            logging.info("The absolute TE threshold is: " + str(threshent_absolute[0]))

            if maxval_absolute >= threshent_absolute[0] and maxval_absolute > 0:
                threshpass_absolute = True
            else:
                threshpass_absolute = False

        elif not weightcalcdata.sigtest:
            threshent_directional = [None, None, None]
            threshent_absolute = [None, None, None]
            threshpass_directional = None
            threshpass_absolute = None
            directionpass_directional = None
            directionpass_absolute = None

        dataline_directional = [
            causevar,
            affectedvar,
            baseval_directional,
            maxval_directional,
            bestdelay_directional,
            delay_index_directional,
            threshent_directional[0],
            threshent_directional[1],
            threshent_directional[2],
            threshpass_directional,
            directionpass_directional,
        ]

        dataline_absolute = [
            causevar,
            affectedvar,
            baseval_absolute,
            maxval_absolute,
            bestdelay_absolute,
            delay_index_absolute,
            threshent_absolute[0],
            threshent_absolute[1],
            threshent_absolute[2],
            threshpass_absolute,
            directionpass_absolute,
        ]

        dataline_directional = (
            dataline_directional
            + proplist_fwd[delay_index_directional]
            + proplist_bwd[delay_index_directional]
            + [milist_fwd[delay_index_directional]]
            + [milist_bwd[delay_index_directional]]
        )
        # Only need to report one but write second one as check

        dataline_absolute = (
            dataline_absolute
            + proplist_fwd[delay_index_absolute]
            + proplist_bwd[delay_index_absolute]
            + [milist_fwd[delay_index_absolute]]
            + [milist_bwd[delay_index_absolute]]
        )

        datalines = [dataline_directional, dataline_absolute]

        logging.info("The corresponding delay is: " + str(bestdelay_directional))
        logging.info("The TE with no delay is: " + str(weightlist[0][0]))

        return datalines

    def calc_surr_te(
        self, weightcalcdata, causevar, affectedvar, box, delay_index, trials
    ):
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

        thresh_causevardata = box[:, weightcalcdata.variables.index(causevar)][
            weightcalcdata.startindex : weightcalcdata.startindex
            + weightcalcdata.testsize
        ]

        thresh_affectedvardata = box[:, weightcalcdata.variables.index(affectedvar)][
            weightcalcdata.startindex
            + delay_index : weightcalcdata.startindex
            + weightcalcdata.testsize
            + delay_index
        ]

        original_causal = np.zeros((1, len(thresh_causevardata)))
        original_causal[0, :] = thresh_causevardata

        if self.surr_method == "iAAFT":
            surr_tsdata = [
                data_processing.gen_iaaft_surrogates(original_causal, 10)
                for n in range(trials)
            ]

        elif self.surr_method == "random_shuffle":
            surr_tsdata = [
                data_processing.shuffle_data(thresh_causevardata) for n in range(trials)
            ]

        surr_te_absolute_list = []
        surr_te_directional_list = []
        for n in range(trials):

            [surr_te_directional, surr_te_absolute], _ = self.calcweight(
                surr_tsdata[n][0, :], thresh_affectedvardata
            )

            surr_te_absolute_list.append(surr_te_absolute)
            surr_te_directional_list.append(surr_te_directional)

        return surr_te_directional_list, surr_te_absolute_list

    def thresh_rankorder(self, surr_te_directional, surr_te_absolute):
        """Calculates the minimum threshold required for a transfer entropy
        value to be considered significant.

        Makes use of a 95% single-sided certainty and a rank-order method.
        This correlates to taking the maximum transfer entropy from 19
        surrogate transfer entropy calculations as the threshold,
        see Schreiber2000a.

        Alternatively, the second highest from 38 observations can be taken,
        etc.

        """

        threshent_directional = max(surr_te_directional)
        nullbias_directional = np.mean(surr_te_directional)
        nullstd_directional = np.std(surr_te_directional)

        threshent_absolute = max(surr_te_absolute)
        nullbias_absolute = np.mean(surr_te_absolute)
        nullstd_absolute = np.std(surr_te_absolute)

        return (
            [threshent_directional, nullbias_directional, nullstd_directional],
            [threshent_absolute, nullbias_absolute, nullstd_absolute],
        )

    def thresh_stdevs(self, surr_te_directional, surr_te_absolute, stdevs):
        """Calculates the minimum threshold required for a transfer entropy
        value to be considered significant.

        Makes use of a six sigma Gaussian check as done in Bauer2005 with 30
        samples of surrogate data.

        """

        surr_te_directional_mean = np.mean(surr_te_directional)
        surr_te_directional_stdev = np.std(surr_te_directional)

        surr_te_absolute_mean = np.mean(surr_te_absolute)
        surr_te_absolute_stdev = np.std(surr_te_absolute)

        threshent_directional = (
            stdevs * surr_te_directional_stdev
        ) + surr_te_directional_mean

        threshent_absolute = (stdevs * surr_te_absolute_stdev) + surr_te_absolute_mean

        return (
            [
                threshent_directional,
                surr_te_directional_mean,
                surr_te_directional_stdev,
            ],
            [threshent_absolute, surr_te_absolute_mean, surr_te_absolute_stdev],
        )

    def calcsigthresh(self, weightcalcdata, causevar, affectedvar, box, delay):

        # This is only called when the entropy at each delay is tested

        if self.thresh_method == "rankorder":
            surr_te_directional, surr_te_absolute = self.calc_surr_te(
                weightcalcdata, causevar, affectedvar, box, delay, 19
            )
            threshent_directional, threshent_absolute = self.thresh_rankorder(
                surr_te_directional, surr_te_absolute
            )
        elif self.thresh_method == "stdevs":
            surr_te_directional, surr_te_absolute = self.calc_surr_te(
                weightcalcdata, causevar, affectedvar, box, delay, 30
            )
            threshent_directional, threshent_absolute = self.thresh_stdevs(
                surr_te_directional, surr_te_absolute, 3
            )

        return [threshent_directional[0], threshent_absolute[0]]
