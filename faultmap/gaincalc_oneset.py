# -*- coding: utf-8 -*-
"""Calculates weight and auxilliary data for each causevar and writes to files.

All weight data file output writers are now called at this level, making the
process interruption tolerant up to a single causevar analysis.

"""

import csv
import logging
import os
from functools import partial

import numpy as np
import pathos
from pathos.multiprocessing import ProcessingPool as Pool


def writecsv_weightcalc(filename, datalines, header):
    """CSV writer customized for writing weights."""

    with open(filename, "w", newline="") as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(datalines)


def readcsv_weightcalc(filename):
    """CSV reader customized for reading weights."""

    with open(filename) as f:
        header = next(csv.reader(f))[:]
        values = np.genfromtxt(f, delimiter=",", dtype=str)

    return values, header


def calc_weights_oneset(
    weightcalcdata,
    weightcalculator,
    box,
    startindex,
    size,
    newconnectionmatrix,
    method,
    boxindex,
    filename,
    headerline,
    writeoutput,
    causevarindex,
):

    causevar = weightcalcdata.variables[causevarindex]

    print(
        "Start analysing causal variable: "
        + causevar
        + " ["
        + str(causevarindex + 1)
        + "/"
        + str(len(weightcalcdata.causevarindexes))
        + "]"
    )

    directional_name = "weights_directional"
    absolute_name = "weights_absolute"
    neutral_name = "weights"

    mis_directional_name = "mis_directional"
    mis_absolute_name = "mis_absolute"
    mis_neutral_name = "mis"

    auxdirectional_name = "auxdata_directional"
    auxabsolute_name = "auxdata_absolute"
    auxneutral_name = "auxdata"

    # Provide names for the significance threshold file types
    if weightcalcdata.allthresh:
        sig_directional_name = "sigthresh_directional"
        sig_absolute_name = "sigthresh_absolute"
        sig_neutral_name = "sigthresh"

    # Initiate datalines with delays
    datalines_directional = np.asarray(weightcalcdata.actual_delays)
    datalines_directional = datalines_directional[:, np.newaxis]
    datalines_absolute = datalines_directional.copy()
    datalines_neutral = datalines_directional.copy()

    # Datalines needed to store mutual information
    mis_datalines_directional = datalines_directional.copy()
    mis_datalines_absolute = datalines_directional.copy()
    mis_datalines_neutral = datalines_directional.copy()

    # Datalines needed to store significance threshold values
    # for each variable combination
    datalines_sigthresh_directional = datalines_directional.copy()
    datalines_sigthresh_absolute = datalines_directional.copy()
    datalines_sigthresh_neutral = datalines_directional.copy()

    # Initiate empty auxdata lists
    auxdata_directional = []
    auxdata_absolute = []
    auxdata_neutral = []

    if method[:16] == "transfer_entropy":
        if os.path.exists(filename(auxdirectional_name, boxindex + 1, causevar)):
            auxdata_directional = list(
                np.genfromtxt(
                    filename(auxdirectional_name, boxindex + 1, causevar),
                    delimiter=",",
                    dtype=str,
                )[1:, :]
            )
            auxdata_absolute = list(
                np.genfromtxt(
                    filename(auxdirectional_name, boxindex + 1, causevar),
                    delimiter=",",
                    dtype=str,
                )[1:, :]
            )

            datalines_directional, _ = readcsv_weightcalc(
                filename(directional_name, boxindex + 1, causevar)
            )

            datalines_absolute, _ = readcsv_weightcalc(
                filename(absolute_name, boxindex + 1, causevar)
            )

            mis_datalines_directional, _ = readcsv_weightcalc(
                filename(mis_directional_name, boxindex + 1, causevar)
            )

            mis_datalines_absolute, _ = readcsv_weightcalc(
                filename(mis_absolute_name, boxindex + 1, causevar)
            )

            if weightcalcdata.allthresh:
                datalines_sigthresh_directional = readcsv_weightcalc(
                    filename(sig_directional_name, boxindex + 1, causevar)
                )
                datalines_sigthresh_absolute = readcsv_weightcalc(
                    filename(sig_absolute_name, boxindex + 1, causevar)
                )

    for affectedvarindex in weightcalcdata.affectedvarindexes:
        affectedvar = weightcalcdata.variables[affectedvarindex]

        logging.info(
            "Analysing effect of: "
            + causevar
            + " on "
            + affectedvar
            + " for box number: "
            + str(boxindex + 1)
        )

        exists = False
        do_test = not (newconnectionmatrix[affectedvarindex, causevarindex] == 0)
        # Test if the affectedvar has already been calculated
        if (method[:16] == "transfer_entropy") and do_test:
            testlocation = filename(auxdirectional_name, boxindex + 1, causevar)
            if os.path.exists(testlocation):
                # Open CSV file and read names of second affected vars
                auxdatafile = np.genfromtxt(
                    testlocation,
                    delimiter=",",
                    usecols=np.arange(0, 2),
                    dtype=str,
                )
                affectedvars = auxdatafile[:, 1]
                if affectedvar in affectedvars:
                    print("Affected variable results in existence")
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

            for delay in weightcalcdata.sample_delays:
                logging.info("Now testing delay: " + str(delay))

                causevardata = box[:, causevarindex][startindex : startindex + size]

                affectedvardata = box[:, affectedvarindex][
                    startindex + delay : startindex + size + delay
                ]

                weight, auxdata = weightcalculator.calc_weight(
                    causevardata,
                    affectedvardata,
                    weightcalcdata,
                    causevarindex,
                    affectedvarindex,
                )

                # Calculate significance thresholds at each data point
                if weightcalcdata.allthresh:
                    sigthreshold = weightcalculator.calc_significance_threshold(
                        weightcalcdata, affectedvar, causevar, box, delay
                    )

                if len(weight) > 1:
                    # If weight contains directional as well as
                    # absolute weights, write to separate lists
                    directional_weightlist.append(weight[0])
                    absolute_weightlist.append(weight[1])
                    # Same approach with significance thresholds
                    if weightcalcdata.allthresh:
                        directional_sigthreshlist.append(sigthreshold[0])
                        absolute_sigthreshlist.append(sigthreshold[1])

                else:
                    weightlist.append(weight[0])
                    if weightcalcdata.allthresh:
                        sigthreshlist.append(sigthreshold[0])

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
                ) = weightcalculator.report(
                    weightcalcdata,
                    causevarindex,
                    affectedvarindex,
                    weightlist,
                    box,
                    proplist,
                    milist,
                )

                auxdata_directional.append(auxdata_thisvar_directional)
                auxdata_absolute.append(auxdata_thisvar_absolute)

                # Do the same for the significance threshold
                if weightcalcdata.allthresh:
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

                auxdata_thisvar_neutral = weightcalculator.report(
                    weightcalcdata,
                    causevarindex,
                    affectedvarindex,
                    weightlist,
                    box,
                    proplist,
                )

                auxdata_neutral.append(auxdata_thisvar_neutral)

                # Write the significance thresholds to file
                if weightcalcdata.allthresh:
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
            not (newconnectionmatrix[affectedvarindex, causevarindex] == 0)
            and (exists is False)
            and (writeoutput is True)
        ):

            if twodimensions:
                writecsv_weightcalc(
                    filename(directional_name, boxindex + 1, causevar),
                    datalines_directional,
                    headerline,
                )

                writecsv_weightcalc(
                    filename(absolute_name, boxindex + 1, causevar),
                    datalines_absolute,
                    headerline,
                )

                # Write mutual information over multiple delays to file just as for transfer entropy
                writecsv_weightcalc(
                    filename(mis_directional_name, boxindex + 1, causevar),
                    mis_datalines_directional,
                    headerline,
                )

                writecsv_weightcalc(
                    filename(mis_absolute_name, boxindex + 1, causevar),
                    mis_datalines_absolute,
                    headerline,
                )

                writecsv_weightcalc(
                    filename(auxdirectional_name, boxindex + 1, causevar),
                    auxdata_directional,
                    weightcalculator.data_header,
                )

                writecsv_weightcalc(
                    filename(auxabsolute_name, boxindex + 1, causevar),
                    auxdata_absolute,
                    weightcalculator.data_header,
                )

                if weightcalcdata.allthresh:
                    writecsv_weightcalc(
                        filename(sig_directional_name, boxindex + 1, causevar),
                        datalines_sigthresh_directional,
                        headerline,
                    )

                    writecsv_weightcalc(
                        filename(sig_absolute_name, boxindex + 1, causevar),
                        datalines_sigthresh_absolute,
                        headerline,
                    )

            else:
                writecsv_weightcalc(
                    filename(neutral_name, boxindex + 1, causevar),
                    datalines_neutral,
                    headerline,
                )

                writecsv_weightcalc(
                    filename(auxneutral_name, boxindex + 1, causevar),
                    auxdata_neutral,
                    weightcalculator.data_header,
                )

                if weightcalcdata.allthresh:
                    writecsv_weightcalc(
                        filename(sig_neutral_name, boxindex + 1, causevar),
                        datalines_sigthresh_neutral,
                        headerline,
                    )

    print(
        "Done analysing causal variable: "
        + causevar
        + " ["
        + str(causevarindex + 1)
        + "/"
        + str(len(weightcalcdata.causevarindexes))
        + "]"
    )

    return None


def run(non_iter_args, do_multiprocessing):
    [
        weightcalcdata,
        weightcalculator,
        box,
        startindex,
        size,
        newconnectionmatrix,
        method,
        boxindex,
        filename,
        headerline,
        writeoutput,
    ] = non_iter_args

    partial_gaincalc_oneset = partial(
        calc_weights_oneset,
        weightcalcdata,
        weightcalculator,
        box,
        startindex,
        size,
        newconnectionmatrix,
        method,
        boxindex,
        filename,
        headerline,
        writeoutput,
    )

    if do_multiprocessing:
        pool = Pool(processes=pathos.multiprocessing.cpu_count())
        pool.map(partial_gaincalc_oneset, weightcalcdata.causevarindexes)

        # Current solution to no close and join methods on ProcessingPool
        # https://github.com/uqfoundation/pathos/issues/46

        s = pathos.multiprocessing.__STATE["pool"]
        s.close()
        s.join()
        pathos.multiprocessing.__STATE["pool"] = None

    else:
        for causevarindex in weightcalcdata.causevarindexes:
            partial_gaincalc_oneset(causevarindex)

    return None
