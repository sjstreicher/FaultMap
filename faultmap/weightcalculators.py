"""This module stores the weight calculator classes used by the weightcalc module."""

import abc
import logging
import types
from typing import TYPE_CHECKING

import numpy as np

from faultmap import data_processing, infodynamics
from faultmap.type_definitions import MutualInformationMethods

if TYPE_CHECKING:
    from faultmap.weightcalc import WeightCalcData

# TODO: Finish refactoring common methods out


def flexiblemethod(method):
    """
    Decorator to allow methods to be defined as either static or instance methods.
    """

    def _flexible_method_wrapper(self, *args, **kwargs):
        if isinstance(method, types.FunctionType):
            return method(*args, **kwargs)
        return method(self, *args, **kwargs)

    return _flexible_method_wrapper


class WeightCalculator(abc.ABC):
    """Abstract base class for weight calculators."""

    def __init__(self, weight_calc_data: "WeightCalcData", *_):
        """Read the files or functions and returns required data fields."""
        self.weight_calc_data = weight_calc_data
        if weight_calc_data.sigtest:
            self.threshold_method = weight_calc_data.threshold_method
            self.surrogate_method = weight_calc_data.surrogate_method

    @flexiblemethod
    @abc.abstractmethod
    def calculate_weight(self, *_):
        """Calculates the weight between two vectors containing timer series data."""
        pass

    def calculate_surrogate_weight(self, *_):
        """Calculates surrogate weights for significance testing."""
        pass

    @abc.abstractmethod
    def calculate_significance_threshold(
        self, source_var: str, destination_var: str, box, delay: int
    ):
        """Calculates the significance threshold for the weight between two vectors
        containing timer series data.

        """
        pass

    @abc.abstractmethod
    def report(self, *_):
        """Calculates and reports the relevant output for each combination
        of variables tested.

        """
        pass


class CorrelationWeightCalculator(WeightCalculator):
    """Implementation of WeightCalculator for correlation-based weight calculation.

    Calculates correlation using covariance with optional standardisation and
    de-trending. This allows the effect of Skogestad scaling to be reflected in final
    result.

    """

    def __init__(self, weight_calc_data: "WeightCalcData"):
        super().__init__(weight_calc_data)

        self.data_header = [
            "source_variable",
            "destination_variable",
            "base_corr",
            "max_corr",
            "max_delay",
            "max_index",
            "sign_change",
            "threshold_correlation",
            "thresh_directionality",
            "significance_threshold_passed",
            "directionality_check_passed",
            "directionality_value",
        ]

    @staticmethod
    def calculate_weight(source_var_data, destination_var_data, *_):
        """Calculates the correlation between two vectors containing
        timer series data.

        """

        # corrval = np.corrcoef(source_var_data.T, affectedvardata.T)[1, 0]
        # TODO: Provide the option of scaling the correlation measure
        # Un-normalised measure
        corrval = np.cov(source_var_data.T, destination_var_data.T)[1, 0]
        # Normalised measure
        # corrval = np.corrcoef(source_var_data.T, affectedvardata.T)[1, 0]
        # Here we use the biased correlation measure
        # corrval = np.correlate(source_var_data.T,
        #     affectedvardata.T)[0] / len(affectedvardata)

        return [corrval], None

    # def calculate_significance_threshold(self, *_):
    #     # Due to the decision to make use of non-standardised correlation, a normal
    #     # surrogate thresholding approach will be used for each individual pair
    #     # These are the Bauer2005 thresholds
    #     # self.threshcorr = (1.85*(self.weight_calc_data.test_size**(-0.41))) + \
    #     #     (2.37*(self.weight_calc_data.test_size**(-0.53)))
    #     # self.threshdir = 0.46*(self.weight_calc_data.test_size**(-0.16))
    #     #        logging.info("Directionality threshold: " + str(self.threshdir))
    #     #        logging.info("Correlation threshold: " + str(self.threshcorr))

    #     return [self.threshcorr]

    def calculate_surrogate_weight(
        self,
        source_var: str,
        destination_var: str,
        box: int,
        trials: int,
    ):
        """Calculates surrogate correlation values for significance
        threshold purposes.

        Two methods for generating surrogate data is available:
        iAAFT (Schreiber 2000a) or random_shuffle in time.

        Returns list of surrogate correlation entropy values of length num.

        """

        # The causal (or source) data is replaced by surrogate data,
        # while the affected (or destination) data remains unchanged.

        # Generate surrogate causal data
        thresh_source_var_data = box[
            :, self.weight_calc_data.variables.index(source_var)
        ][
            self.weight_calc_data.start_index : self.weight_calc_data.start_index
            + self.weight_calc_data.test_size
        ]

        # Get the causal data in the correct format
        # for surrogate generation
        original_causal = np.zeros((1, len(thresh_source_var_data)))
        original_causal[0, :] = thresh_source_var_data

        if self.surrogate_method == "iAAFT":
            surr_tsdata = [
                data_processing.gen_iaaft_surrogates(original_causal, 10)
                for _ in range(trials)
            ]
        elif self.surrogate_method == "random_shuffle":
            surr_tsdata = [
                data_processing.shuffle_data(thresh_source_var_data)
                for _ in range(trials)
            ]

        surr_corr_list = []
        surr_dirindex_list = []
        for n in range(trials):
            # Compute the weightlist for every trial by evaluating all delays
            surrogate_weight_list = []
            for delay_index in self.weight_calc_data.sample_delays:
                threshold_destination_var_data = box[
                    :, self.weight_calc_data.variables.index(destination_var)
                ][
                    self.weight_calc_data.start_index
                    + delay_index : self.weight_calc_data.start_index
                    + self.weight_calc_data.test_size
                    + delay_index
                ]
                surrogate_weight_list.append(
                    self.calculate_weight(
                        surr_tsdata[n][0, :], threshold_destination_var_data
                    )[0][0]
                )

            _, maxcorr, _, _, _, directionindex, _ = self.select_weights(
                source_var, destination_var, surrogate_weight_list
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

    def calculate_significance_threshold(self, source_var, destination_var, box, _):
        # Calculates correlation as well as correlation directionality index
        # significance thresholds
        # The correlation threshold is a simple statistical significance test
        # based on surrogate results
        # The directionality threshold calculation is as described by Bauer2005

        # This is only called when the entropy at each delay is tested

        if self.threshold_method == "rankorder":
            surr_corr, surr_dirindex = self.calculate_surrogate_weight(
                source_var, destination_var, box, 19
            )
            thresh_corr, thresh_dirindex = self.thresh_rankorder(
                surr_corr, surr_dirindex
            )
        elif self.threshold_method == "stdevs":
            surr_corr, surr_dirindex = self.calculate_surrogate_weight(
                source_var, destination_var, box, 30
            )
            thresh_corr, thresh_dirindex = self.thresh_stdevs(
                surr_corr, surr_dirindex, 3
            )
        else:
            raise ValueError("Threshold method not recognized")

        return [thresh_corr[0], thresh_dirindex[0]]

    def select_weights(self, source_var, destination_var, weights: list):
        if self.weight_calc_data.bidirectional_delays:
            base_value = weights[int(len(weights) / 2)]
        else:
            base_value = weights[0]

        # Alternative interpretation:
        # max_val is defined as the maximum absolute value in the forward direction
        # min_val is defined as the maximum absolute value in the negative direction
        if self.weight_calc_data.bidirectional_delays:
            max_val = max(np.abs(weights[int(len(weights) / 2) :]))
            min_val = -1 * max(np.abs(weights[: int(len(weights) / 2)]))
        else:
            raise ValueError(
                "The correlation directionality test is only defined for bidirectional "
                "delays"
            )

        # Test that maxval is positive and min_val is negative
        if max_val < 0 or min_val > 0:
            raise ValueError("Values do not adhere to sign expectations")
        # Value used to break tie between maxval and min_val if 1 and -1
        tol = 0.0
        # Always select maxval if both are equal
        # This is to ensure that 1 is selected above -1
        if (max_val + (min_val + tol)) >= 0:
            max_corr = max_val
        else:
            max_corr = abs(min_val)

        delay_index = list(np.abs(weights)).index(max_corr)

        # Correlation thresholds from Bauer2008 Eq. 4
        # max_corr_abs = abs(max_corr)
        best_delay = self.weight_calc_data.actual_delays[delay_index]
        if (max_val and min_val) != 0:
            direction_index = abs(max_val + min_val) / ((max_val + abs(min_val)) * 0.5)
        else:
            direction_index = 0

        sign_change = not ((base_value / weights[delay_index]) >= 0)

        logging.info("Maximum forward correlation value: %s", str(max_val))
        logging.info("Maximum backward correlation value: %s", str(min_val))
        logging.info(
            "The maximum correlation between %s and %s is: %s",
            source_var,
            destination_var,
            str(max_corr),
        )
        logging.info("The corresponding delay is: %s", str(best_delay))
        logging.info("The correlation with no delay is: %s", str(base_value))

        logging.info("Directionality value: %s", str(direction_index))

        best_delay_sample = self.weight_calc_data.sample_delays[delay_index]

        return (
            base_value,
            max_corr,
            delay_index,
            best_delay,
            best_delay_sample,
            direction_index,
            sign_change,
        )

    def report(
        self,
        source_var_index,
        destination_var_index,
        weight_list,
        box,
        _,
    ):
        """Calculates and reports the relevant output for each combination
        of variables tested.

        """
        source_var = self.weight_calc_data.variables[source_var_index]
        destination_var = self.weight_calc_data.variables[destination_var_index]

        # Get best weights and delays and an indication whether the directionality test
        # was passed

        (
            baseval,
            maxcorr,
            delay_index,
            bestdelay,
            bestdelay_sample,
            directionindex,
            signchange,
        ) = self.select_weights(source_var, destination_var, weight_list)

        corrthreshpass = None
        dirthreshpass = None

        if self.weight_calc_data.sigtest:
            # Calculate value and directionality thresholds for correlation

            # In the case of correlation, we need to create one surrogate weightlist
            # (evaluation at all delays) for every trial, and get the maxcorr and
            # directionality index from it.

            # Do significance calculations

            if self.threshold_method == "rankorder":
                surr_corr, surr_dirindex = self.calculate_surrogate_weight(
                    source_var, destination_var, box, 19
                )
                threshcorr, threshdir = self.thresh_rankorder(surr_corr, surr_dirindex)
            elif self.threshold_method == "stdevs":
                surr_corr, surr_dirindex = self.calculate_surrogate_weight(
                    source_var, destination_var, box, 30
                )
                threshcorr, threshdir = self.thresh_stdevs(surr_corr, surr_dirindex, 3)

            logging.info("The correlation threshold is: " + str(threshcorr[0]))
            logging.info("The direction index threshold is: " + str(threshdir[0]))

            corrthreshpass = abs(maxcorr) >= threshcorr[0]
            dirthreshpass = (directionindex >= threshdir[0]) and (bestdelay >= 0.0)
            logging.info("Correlation threshold passed: " + str(corrthreshpass))
            logging.info("Directionality threshold passed: " + str(dirthreshpass))

        elif not self.weight_calc_data.sigtest:
            threshcorr = [None]
            threshdir = [None]

        # The maxcorr value can be positive or negative, the eigenvector faultmap steps
        # needs to abs all the weights

        dataline = [
            source_var,
            destination_var,
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


class TransferEntropyWeightCalculator(WeightCalculator):
    """Transfer entropy based weight calculation."""

    def __init__(
        self, weight_calc_data: "WeightCalcData", estimator: MutualInformationMethods
    ):
        super().__init__(weight_calc_data)
        self.data_header = [
            "source_variable",
            "destination_variable",
            "base_ent",
            "max_ent",
            "max_delay",
            "max_index",
            "threshold",
            "bias_mean",
            "bias_std",
            "significance_threshold_passed",
            "directionality_check_passed",
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

        self.parameters = {}
        self.estimator = estimator
        self.infodynamics_location = weight_calc_data.infodynamics_loc

        if self.estimator == "kraskov":
            parameters_dict = weight_calc_data.additional_parameters
            parameters_dict["use_gpu"] = weight_calc_data.use_gpu
            self.parameters = weight_calc_data.additional_parameters

        # Add kernel bandwidth to parameters
        if (self.estimator == "kernel") and (weight_calc_data.kernel_width is not None):
            self.parameters["kernel_width"] = weight_calc_data.kernel_width

    def calculate_weight(self, cause_var_data, affected_var_data, *_):
        """ "Calculates the transfer entropy between two vectors containing
        timer series data.

        """
        # Calculate directional transfer entropy as the difference
        # between the forward and backwards entropy

        # Pass special estimator specific parameters in here

        transfer_entropy_forward, aux_data_forward = infodynamics.calc_te(
            self.infodynamics_location,
            self.estimator,
            affected_var_data.T,
            cause_var_data.T,
            **self.parameters,
        )

        transent_bwd, auxdata_bwd = infodynamics.calc_te(
            self.infodynamics_location,
            self.estimator,
            cause_var_data.T,
            affected_var_data.T,
            **self.parameters,
        )

        transent_directional = transfer_entropy_forward - transent_bwd
        transent_absolute = transfer_entropy_forward

        return (
            [transent_directional, transent_absolute],
            [aux_data_forward, auxdata_bwd],
        )

    def select_weights(
        self,
        source_var: str,
        destination_var: str,
        weight_list: list,
        directional: bool,
    ):
        if directional:
            directionality = "directional"
        else:
            directionality = "absolute"
        # Initiate flag indicating whether direction test passed
        pass_directionality = None

        if self.weight_calc_data.bidirectional_delays:
            base_val = weight_list[int(len(weight_list) / 2)]

            # Get maximum weight in forward direction
            # This includes all positive delays including zero
            max_val_forward = max(weight_list[int((len(weight_list) - 1) / 2) :])
            # Get maximum weight in backward direction
            # This includes all negative delays excluding zero
            max_val_backward = max(weight_list[: int((len(weight_list) - 1) / 2)])

            delay_index_forward = weight_list.index(max_val_forward)
            delay_index_backward = weight_list.index(max_val_backward)

            best_forward_delay = self.weight_calc_data.actual_delays[
                delay_index_forward
            ]
            best_backward_delay = self.weight_calc_data.actual_delays[
                delay_index_backward
            ]

            # Test if the maximum forward value is bigger than the maximum
            # backward value
            if max_val_forward > max_val_backward:
                # We accept this value as true (pending significance testing)
                pass_directionality = True
            else:
                # Test whether the maximum forward delay is smaller than the
                # maximum backward delay
                # If this test passes we still consider the direction test to
                # pass
                if best_forward_delay < abs(best_backward_delay):
                    pass_directionality = True
                else:
                    pass_directionality = False

            # Assign forward values to generic outputs
            max_val = max_val_forward
            delay_index = delay_index_forward
            best_delay = best_forward_delay

            logging.info(
                "The maximum forward "
                + directionality
                + " TE between "
                + source_var
                + " and "
                + destination_var
                + " is: "
                + str(max_val_forward)
            )
            logging.info(
                "The maximum backward "
                + directionality
                + " TE between "
                + source_var
                + " and "
                + destination_var
                + " is: "
                + str(max_val_backward)
            )
            if pass_directionality is True:
                logging.info("The directionality test passed")
            elif pass_directionality is False:
                logging.info("The directionality test failed")

        else:
            base_val = weight_list[0]
            max_val = max(weight_list)
            delay_index = weight_list.index(max_val)
            best_delay = self.weight_calc_data.actual_delays[delay_index]

            logging.info(
                "The maximum "
                + directionality
                + " TE between "
                + source_var
                + " and "
                + destination_var
                + " is: "
                + str(max_val)
            )

        bestdelay_sample = self.weight_calc_data.sample_delays[delay_index]

        return (
            base_val,
            max_val,
            delay_index,
            best_delay,
            bestdelay_sample,
            pass_directionality,
        )

    def report(
        self,
        source_var_index,
        destination_var_index,
        weight_list,
        box,
        proplist,
        milist,
    ):
        """Calculates and reports the relevant output for each combination
        of variables tested.

        """

        variables = self.weight_calc_data.variables
        source_var = variables[source_var_index]
        destination_var = variables[destination_var_index]

        # We already know that when dealing with transfer entropy
        # the weightlist will consist of a list of lists
        weight_list_directional, weight_list_absolute = weight_list

        proplist_fwd, proplist_bwd = proplist
        milist_fwd, milist_bwd = milist

        # Not supposed to differ among delay test
        # TODO: Confirm this
        #        k_hist_fwd, k_tau_fwd, l_hist_fwd, l_tau_fwd, delay_fwd = \
        #            proplist_fwd[0]
        #        k_hist_bwd, k_tau_bwd, l_hist_bwd, l_tau_bwd, delay_bwd = \
        #            proplist_bwd[0]

        size = self.weight_calc_data.test_size
        start_index = self.weight_calc_data.start_index

        # Get best weights and delays and an indication whether the
        # directionality test was passed for bidirectional testing cases

        # Do everything for the directional case
        (
            baseval_directional,
            max_val_directional,
            delay_index_directional,
            best_delay_directional,
            best_sample_delay_directional,
            directionpass_directional,
        ) = self.select_weights(
            source_var, destination_var, weight_list_directional, True
        )

        # Repeat for the absolute case
        (
            baseval_absolute,
            maxval_absolute,
            delay_index_absolute,
            bestdelay_absolute,
            bestdelay_sample_absolute,
            directionpass_absolute,
        ) = self.select_weights(
            source_var, destination_var, weight_list_absolute, False
        )

        if self.weight_calc_data.sigtest:
            # Calculate threshold for transfer entropy
            thresh_source_var_data = box[  # noqa: F841
                :, source_var_index
            ][start_index : start_index + size]
            thresh_affectedvardata_directional = box[  # noqa: F841
                :, destination_var_index
            ][
                start_index + best_sample_delay_directional : start_index
                + size
                + best_sample_delay_directional
            ]

            thresh_affectedvardata_absolute = box[  # noqa: F841
                :, destination_var_index
            ][
                start_index + bestdelay_sample_absolute : start_index
                + size
                + bestdelay_sample_absolute
            ]

            # Do significance calculations for directional case
            if self.threshold_method == "rankorder":
                surr_te_directional, surr_te_absolute = self.calculate_surrogate_weight(
                    source_var,
                    destination_var,
                    box,
                    best_sample_delay_directional,
                    19,
                )
                (
                    threshent_directional,
                    threshent_absolute,
                ) = self.threshold_rankorder(surr_te_directional, surr_te_absolute)
            elif self.threshold_method == "stdevs":
                surr_te_directional, surr_te_absolute = self.calculate_surrogate_weight(
                    source_var,
                    destination_var,
                    box,
                    best_sample_delay_directional,
                    30,
                )
                threshent_directional, threshent_absolute = self.thresh_stdevs(
                    surr_te_directional, surr_te_absolute, 3
                )

            logging.info(
                "The directional TE threshold is: " + str(threshent_directional[0])
            )

            if (
                max_val_directional >= threshent_directional[0]
                and max_val_directional > 0
            ):
                threshpass_directional = True
            else:
                threshpass_directional = False

            if not delay_index_directional == delay_index_absolute:
                # Need to do own calculation of absolute significance
                if self.threshold_method == "rankorder":
                    (
                        surr_te_directional,
                        surr_te_absolute,
                    ) = self.calculate_surrogate_weight(
                        source_var,
                        destination_var,
                        box,
                        bestdelay_sample_absolute,
                        19,
                    )
                    _, threshent_absolute = self.threshold_rankorder(
                        surr_te_directional, surr_te_absolute
                    )
                elif self.threshold_method == "stdevs":
                    (
                        surr_te_directional,
                        surr_te_absolute,
                    ) = self.calculate_surrogate_weight(
                        source_var,
                        destination_var,
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

        elif not self.weight_calc_data.sigtest:
            threshent_directional = [None, None, None]
            threshent_absolute = [None, None, None]
            threshpass_directional = None
            threshpass_absolute = None
            directionpass_directional = None
            directionpass_absolute = None

        dataline_directional = [
            source_var,
            destination_var,
            baseval_directional,
            max_val_directional,
            best_delay_directional,
            delay_index_directional,
            threshent_directional[0],
            threshent_directional[1],
            threshent_directional[2],
            threshpass_directional,
            directionpass_directional,
        ]

        dataline_absolute = [
            source_var,
            destination_var,
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

        logging.info("The corresponding delay is: " + str(best_delay_directional))
        logging.info("The TE with no delay is: " + str(weight_list[0][0]))

        return datalines

    def calculate_surrogate_weight(
        self, source_var, destination_var, box, delay_index: int, trials: int
    ):
        """Calculates surrogate transfer entropy values for significance
        threshold purposes.

        Two methods for generating surrogate data is available:
        iAAFT (Schreiber 2000a) or random_shuffle in time.

        Returns list of surrogate transfer entropy values of length num.

        """

        # The causal (or source) data is replaced by surrogate data,
        # while the affected (or destination) data remains unchanged.

        # Get the source data in the correct format for surrogate generation

        thresh_source_var_data = box[
            :, self.weight_calc_data.variables.index(source_var)
        ][
            self.weight_calc_data.start_index : self.weight_calc_data.start_index
            + self.weight_calc_data.test_size
        ]

        thresh_affectedvardata = box[
            :, self.weight_calc_data.variables.index(destination_var)
        ][
            self.weight_calc_data.start_index
            + delay_index : self.weight_calc_data.start_index
            + self.weight_calc_data.test_size
            + delay_index
        ]

        # TODO: Review whyt this is done
        original_source = np.zeros((1, len(thresh_source_var_data)))
        original_source[0, :] = thresh_source_var_data

        if self.surrogate_method == "iAAFT":
            surr_tsdata = [
                data_processing.gen_iaaft_surrogates(original_source, 10)
                for n in range(trials)
            ]

        elif self.surrogate_method == "random_shuffle":
            surr_tsdata = [
                data_processing.shuffle_data(thresh_source_var_data)
                for n in range(trials)
            ]

        surr_te_absolute_list = []
        surr_te_directional_list = []
        for n in range(trials):
            [surr_te_directional, surr_te_absolute], _ = self.calculate_weight(
                surr_tsdata[n][0, :], thresh_affectedvardata
            )

            surr_te_absolute_list.append(surr_te_absolute)
            surr_te_directional_list.append(surr_te_directional)

        return surr_te_directional_list, surr_te_absolute_list

    @staticmethod
    def threshold_rankorder(surrogate_directional_weights, surrogate_absolute_weights):
        """Calculates the minimum threshold required for a transfer entropy
        value to be considered significant.

        Makes use of a 95% single-sided certainty and a rank-order method.
        This correlates to taking the maximum transfer entropy from 19
        surrogate transfer entropy calculations as the threshold,
        see Schreiber2000a.

        Alternatively, the second highest from 38 observations can be taken,
        etc.

        """

        threshent_directional = max(surrogate_directional_weights)
        nullbias_directional = np.mean(surrogate_directional_weights)
        nullstd_directional = np.std(surrogate_directional_weights)

        threshent_absolute = max(surrogate_absolute_weights)
        nullbias_absolute = np.mean(surrogate_absolute_weights)
        nullstd_absolute = np.std(surrogate_absolute_weights)

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
            [
                threshent_absolute,
                surr_te_absolute_mean,
                surr_te_absolute_stdev,
            ],
        )

    def calculate_significance_threshold(self, source_var, destination_var, box, delay):
        # This is only called when the entropy at each delay is tested

        if self.threshold_method == "rankorder":
            surr_te_directional, surr_te_absolute = self.calculate_surrogate_weight(
                source_var, destination_var, box, delay, 19
            )
            threshent_directional, threshent_absolute = self.threshold_rankorder(
                surr_te_directional, surr_te_absolute
            )
        elif self.threshold_method == "stdevs":
            surr_te_directional, surr_te_absolute = self.calculate_surrogate_weight(
                source_var, destination_var, box, delay, 30
            )
            threshent_directional, threshent_absolute = self.thresh_stdevs(
                surr_te_directional, surr_te_absolute, 3
            )

        return [threshent_directional[0], threshent_absolute[0]]
