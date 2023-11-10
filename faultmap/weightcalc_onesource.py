"""Calculates weight and auxiliary data for each source variable and writes to files.

All weight data file output writers are now called at this level, making the
process interruption tolerant up to a single source variable analysis.

"""

import csv
import logging
import os
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pathos  # type: ignore
from pathos.multiprocessing import ProcessingPool as Pool  # type: ignore

if TYPE_CHECKING:
    from faultmap.weightcalc import WeightCalcData
    from faultmap.weightcalculators import WeightCalculator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def writecsv_weightcalc(filename, datalines, header):
    """CSV writer customized for writing weights."""

    with open(filename, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(datalines)


def readcsv_weightcalc(filename):
    """CSV reader customized for reading weights."""

    with open(filename, encoding="utf-8") as f:
        header = next(csv.reader(f))[:]
        values = np.genfromtxt(f, delimiter=",", dtype=str)

    return values, header


def calc_weights_one_source(
    weight_calc_data: "WeightCalcData",
    weight_calculator: "WeightCalculator",
    box,
    start_index,
    size,
    new_connection_matrix,
    method,
    box_index: int,
    filename,
    header_line,
    write_output: bool,
    source_var_index: int,
):
    source_var = weight_calc_data.variables[source_var_index]

    logger.info(
        "Start analysing source variable: "
        + source_var
        + " ["
        + str(source_var_index + 1)
        + "/"
        + str(len(weight_calc_data.source_var_indexes))
        + "]"
    )

    directional_weights_name = "weights_directional"
    absolute_weights_name = "weights_absolute"
    neutral_weights_name = "weights"

    directional_mi_name = "mis_directional"
    absolute_mi_name = "mis_absolute"

    directional_aux_name = "auxdata_directional"
    absolute_aux_name = "auxdata_absolute"
    neutral_aux_name = "auxdata"

    # Provide names for the significance threshold file types
    if weight_calc_data.all_thresh:
        sig_directional_name = "sigthresh_directional"
        sig_absolute_name = "sigthresh_absolute"
        sig_neutral_name = "sigthresh"

    # Initiate datalines with delays
    datalines_directional = np.asarray(weight_calc_data.actual_delays)
    datalines_directional = datalines_directional[:, np.newaxis]
    datalines_absolute = datalines_directional.copy()
    datalines_neutral = datalines_directional.copy()

    # Datalines needed to store mutual information
    mis_datalines_directional = datalines_directional.copy()
    mis_datalines_absolute = datalines_directional.copy()

    # Datalines needed to store significance threshold values
    # for each variable combination
    datalines_sigthresh_directional = datalines_directional.copy()
    datalines_sigthresh_absolute = datalines_directional.copy()
    datalines_sigthresh_neutral = datalines_directional.copy()

    # Initiate empty auxdata lists
    auxdata_directional = []
    auxdata_absolute = []
    auxdata_neutral = []

    if "transfer_entropy" in method:
        if os.path.exists(filename(directional_aux_name, box_index + 1, source_var)):
            auxdata_directional = list(
                np.genfromtxt(
                    filename(directional_aux_name, box_index + 1, source_var),
                    delimiter=",",
                    dtype=str,
                )[1:, :]
            )
            auxdata_absolute = list(
                np.genfromtxt(
                    filename(directional_aux_name, box_index + 1, source_var),
                    delimiter=",",
                    dtype=str,
                )[1:, :]
            )

            datalines_directional, _ = readcsv_weightcalc(
                filename(directional_weights_name, box_index + 1, source_var)
            )

            datalines_absolute, _ = readcsv_weightcalc(
                filename(absolute_weights_name, box_index + 1, source_var)
            )

            mis_datalines_directional, _ = readcsv_weightcalc(
                filename(directional_mi_name, box_index + 1, source_var)
            )

            mis_datalines_absolute, _ = readcsv_weightcalc(
                filename(absolute_mi_name, box_index + 1, source_var)
            )

            if weight_calc_data.all_thresh:
                datalines_sigthresh_directional = readcsv_weightcalc(
                    filename(sig_directional_name, box_index + 1, source_var)
                )
                datalines_sigthresh_absolute = readcsv_weightcalc(
                    filename(sig_absolute_name, box_index + 1, source_var)
                )

    for destination_var_index in weight_calc_data.destination_var_indexes:
        destination_var = weight_calc_data.variables[destination_var_index]

        logger.info(
            "Analysing effect of: "
            + source_var
            + " on "
            + destination_var
            + " for box number: "
            + str(box_index + 1)
        )

        exists = False
        do_test = not (
            new_connection_matrix[destination_var_index, source_var_index] == 0
        )
        # Test if the affectedvar has already been calculated
        if "transfer_entropy" in method and do_test:
            test_location = filename(directional_aux_name, box_index + 1, source_var)
            if os.path.exists(test_location):
                # Open CSV file and read names of second affected vars
                aux_data_file = np.genfromtxt(
                    test_location,
                    delimiter=",",
                    usecols=np.arange(0, 2),
                    dtype=str,
                )
                destination_vars = aux_data_file[:, 1]
                if destination_var in destination_vars:
                    print("Destination variable results in existence")
                    exists = True

        if do_test and (exists is False):
            weightlist = []
            directional_weightlist = []
            absolute_weightlist = []
            sigthreshlist = []
            directional_sigthreshlist = []
            absolute_sigthreshlist = []
            sigfwd_list = []
            sigbwd_list = []
            propfwd_list = []
            propbwd_list = []
            mifwd_list = []
            mibwd_list = []

            for delay in weight_calc_data.sample_delays:
                logger.info("Now testing delay: %s", str(delay))

                if start_index + delay < 0:
                    raise ValueError(
                        "Start index must be larger than biggest negative delay"
                    )

                source_var_data = box[:, source_var_index][
                    start_index : start_index + size
                ]

                destination_var_data = box[:, destination_var_index][
                    start_index + delay : start_index + size + delay
                ]

                weight, auxdata = weight_calculator.calculate_weight(
                    source_var_data,
                    destination_var_data,
                    weight_calc_data,
                    source_var_index,
                    destination_var_index,
                )

                # Calculate significance thresholds at each data point
                if weight_calc_data.all_thresh:
                    significance_threshold = (
                        # TODO: Follow up on how name order got swapped around
                        #  It is possible that the order of the significance test was
                        #  accidentally swapped around in the original code
                        weight_calculator.calculate_significance_threshold(
                            source_var, destination_var, box, delay
                        )
                    )

                if len(weight) > 1:
                    # If weight contains directional as well as
                    # absolute weights, write to separate lists
                    directional_weightlist.append(weight[0])
                    absolute_weightlist.append(weight[1])
                    # Same approach with significance thresholds
                    if weight_calc_data.all_thresh:
                        directional_sigthreshlist.append(significance_threshold[0])
                        absolute_sigthreshlist.append(significance_threshold[1])

                else:
                    weightlist.append(weight[0])
                    if weight_calc_data.all_thresh:
                        sigthreshlist.append(significance_threshold[0])

                if auxdata is not None:
                    if len(auxdata) > 1:
                        # This means we have auxdata for both the forward and
                        # backward calculation
                        [auxdata_fwd, auxdata_bwd] = auxdata
                        [
                            significance_fwd,
                            properties_fwd,
                            mi_fwd,
                        ] = auxdata_fwd  # mi_fwd and mi_bwd should be the same
                        [
                            significance_bwd,
                            properties_bwd,
                            mi_bwd,
                        ] = auxdata_bwd
                        sigfwd_list.append(significance_fwd)
                        sigbwd_list.append(significance_bwd)
                        propfwd_list.append(properties_fwd)
                        propbwd_list.append(properties_bwd)
                        mifwd_list.append(mi_fwd)
                        mibwd_list.append(mi_bwd)

            if len(weight) > 1:
                twodimensions = True

                proplist = [propfwd_list, propbwd_list]
                milist = [mifwd_list, mibwd_list]
                siglist = [sigfwd_list, sigbwd_list]
                weightlist = [directional_weightlist, absolute_weightlist]

                # Combine weight data
                weights_thisvar_directional = np.asarray(weightlist[0])
                weights_thisvar_directional = weights_thisvar_directional[:, np.newaxis]

                mis_thisvar_directional = np.asarray(milist[0])
                mis_thisvar_directional = mis_thisvar_directional[:, np.newaxis]

                datalines_directional = np.concatenate(
                    (datalines_directional, weights_thisvar_directional),
                    axis=1,
                )

                mis_datalines_directional = np.concatenate(
                    (mis_datalines_directional, mis_thisvar_directional),
                    axis=1,
                )

                weights_thisvar_absolute = np.asarray(weightlist[1])
                weights_thisvar_absolute = weights_thisvar_absolute[:, np.newaxis]

                mis_thisvar_absolute = np.asarray(milist[1])
                mis_thisvar_absolute = mis_thisvar_absolute[:, np.newaxis]

                datalines_absolute = np.concatenate(
                    (datalines_absolute, weights_thisvar_absolute), axis=1
                )

                mis_datalines_absolute = np.concatenate(
                    (mis_datalines_absolute, mis_thisvar_absolute), axis=1
                )

                # Write all the auxiliary weight data
                # Generate and store report files according to each method
                (
                    auxdata_thisvar_directional,
                    auxdata_thisvar_absolute,
                ) = weight_calculator.report(
                    source_var_index,
                    destination_var_index,
                    weightlist,
                    box,
                    proplist,
                    milist,
                )

                auxdata_directional.append(auxdata_thisvar_directional)
                auxdata_absolute.append(auxdata_thisvar_absolute)

                # Do the same for the significance threshold
                if weight_calc_data.all_thresh:
                    sigthreshlist = [
                        directional_sigthreshlist,
                        absolute_sigthreshlist,
                    ]

                    sigthresh_thisvar_directional = np.asarray(sigthreshlist[0])
                    sigthresh_thisvar_directional = sigthresh_thisvar_directional[
                        :, np.newaxis
                    ]

                    datalines_sigthresh_directional = np.concatenate(
                        (
                            datalines_sigthresh_directional,
                            sigthresh_thisvar_directional,
                        ),
                        axis=1,
                    )

                    sigthresh_thisvar_absolute = np.asarray(sigthreshlist[1])
                    sigthresh_thisvar_absolute = sigthresh_thisvar_absolute[
                        :, np.newaxis
                    ]

                    datalines_sigthresh_absolute = np.concatenate(
                        (
                            datalines_sigthresh_absolute,
                            sigthresh_thisvar_absolute,
                        ),
                        axis=1,
                    )

            else:
                twodimensions = False

                weights_thisvar_neutral = np.asarray(weightlist)
                weights_thisvar_neutral = weights_thisvar_neutral[:, np.newaxis]

                datalines_neutral = np.concatenate(
                    (datalines_neutral, weights_thisvar_neutral), axis=1
                )

                # Write all the auxilliary weight data
                # Generate and store report files according to each method
                proplist = None

                auxdata_thisvar_neutral = weight_calculator.report(
                    source_var_index,
                    destination_var_index,
                    weightlist,
                    box,
                    proplist,
                )

                auxdata_neutral.append(auxdata_thisvar_neutral)

                # Write the significance thresholds to file
                if weight_calc_data.all_thresh:
                    sigthresh_thisvar_neutral = np.asarray(sigthreshlist)
                    sigthresh_thisvar_neutral = sigthresh_thisvar_neutral[:, np.newaxis]

                    datalines_sigthresh_neutral = np.concatenate(
                        (
                            datalines_sigthresh_neutral,
                            sigthresh_thisvar_neutral,
                        ),
                        axis=1,
                    )

        if (
            not (new_connection_matrix[destination_var_index, source_var_index] == 0)
            and (exists is False)
            and (write_output is True)
        ):
            if twodimensions:
                writecsv_weightcalc(
                    filename(directional_weights_name, box_index + 1, source_var),
                    datalines_directional,
                    header_line,
                )

                writecsv_weightcalc(
                    filename(absolute_weights_name, box_index + 1, source_var),
                    datalines_absolute,
                    header_line,
                )

                # Write mutual information over multiple delays to file just as for transfer entropy
                writecsv_weightcalc(
                    filename(directional_mi_name, box_index + 1, source_var),
                    mis_datalines_directional,
                    header_line,
                )

                writecsv_weightcalc(
                    filename(absolute_mi_name, box_index + 1, source_var),
                    mis_datalines_absolute,
                    header_line,
                )

                writecsv_weightcalc(
                    filename(directional_aux_name, box_index + 1, source_var),
                    auxdata_directional,
                    weight_calculator.data_header,
                )

                writecsv_weightcalc(
                    filename(absolute_aux_name, box_index + 1, source_var),
                    auxdata_absolute,
                    weight_calculator.data_header,
                )

                if weight_calc_data.all_thresh:
                    writecsv_weightcalc(
                        filename(sig_directional_name, box_index + 1, source_var),
                        datalines_sigthresh_directional,
                        header_line,
                    )

                    writecsv_weightcalc(
                        filename(sig_absolute_name, box_index + 1, source_var),
                        datalines_sigthresh_absolute,
                        header_line,
                    )

            else:
                writecsv_weightcalc(
                    filename(neutral_weights_name, box_index + 1, source_var),
                    datalines_neutral,
                    header_line,
                )

                writecsv_weightcalc(
                    filename(neutral_aux_name, box_index + 1, source_var),
                    auxdata_neutral,
                    weight_calculator.data_header,
                )

                if weight_calc_data.all_thresh:
                    writecsv_weightcalc(
                        filename(sig_neutral_name, box_index + 1, source_var),
                        datalines_sigthresh_neutral,
                        header_line,
                    )

    print(
        "Done analysing causal variable: "
        + source_var
        + " ["
        + str(source_var_index + 1)
        + "/"
        + str(len(weight_calc_data.source_var_indexes))
        + "]"
    )


def run(non_iter_args, do_multiprocessing):
    [
        weight_calc_data,
        weight_calculator,
        box,
        start_index,
        size,
        new_connection_matrix,
        method,
        box_index,
        filename,
        header_line,
        write_output,
    ] = non_iter_args

    partial_weightcalc_one_source = partial(
        calc_weights_one_source,
        weight_calc_data,
        weight_calculator,
        box,
        start_index,
        size,
        new_connection_matrix,
        method,
        box_index,
        filename,
        header_line,
        write_output,
    )

    if do_multiprocessing:
        pool = Pool(processes=pathos.multiprocessing.cpu_count())
        pool.map(partial_weightcalc_one_source, weight_calc_data.source_var_indexes)

        # Current solution to no close and join methods on ProcessingPool
        # https://github.com/uqfoundation/pathos/issues/46

        s = pathos.multiprocessing.__STATE["pool"]
        s.close()
        s.join()
        pathos.multiprocessing.__STATE["pool"] = None

    else:
        for source_var_index in weight_calc_data.source_var_indexes:
            partial_weightcalc_one_source(source_var_index)

    return None
