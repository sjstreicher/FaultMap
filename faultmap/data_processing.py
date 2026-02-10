"""Data processing support tasks."""

import csv
import gc
import json
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sklearn.preprocessing  # type: ignore
import tables as tb  # type: ignore
from numba import jit  # type: ignore
from numpy.typing import NDArray
from scipy import signal  # type: ignore

from faultmap import config_setup, infodynamics, weightcalc
from faultmap.type_definitions import EntropyMethods, RunModes
from faultmap.weightcalc import WeightCalcData

logger = logging.getLogger(__name__)

# pylint: disable=too-many-lines, missing-function-docstring


@jit(nopython=True)
def shuffle_data(input_data: NDArray) -> NDArray:
    """Returns a (seeded) randomly shuffled array of data.
    The data input needs to be a two-dimensional numpy array.

    """

    shuffled = np.random.permutation(input_data)

    shuffled_formatted = np.zeros((1, len(shuffled)))
    shuffled_formatted[0, :] = shuffled

    return shuffled_formatted


def get_folders(path: str | Path) -> list[str]:
    folders = []
    while True:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)

            break

    folders.reverse()

    return folders


# Use jit for loop-jitting
@jit(forceobj=True)
def gen_iaaft_surrogates(data: NDArray, iterations: int) -> NDArray:
    """Generates iterative amplitude adjusted Fourier transform (IAAFT) surrogates"""
    # Make copy to prevent rotation of array
    original_data = data.copy()
    start_time = time.process_time()
    # amplitude stored
    sorted_data = original_data.copy()
    sorted_data.sort()
    # amplitude of Fourier transform of original data
    frequency_amplitude = np.abs(np.fft.fft(original_data))

    original_data.shape = (-1, 1)
    # random permutation as starting point
    surrogate_data = np.random.permutation(original_data)
    surrogate_data.shape = (1, -1)

    for _ in range(iterations):
        surrogate_fft = frequency_amplitude * np.exp(
            1j * np.angle(np.fft.fft(surrogate_data))
        )
        xoutb = np.real(np.fft.ifft(surrogate_fft))
        ranks = xoutb.argsort(axis=1)
        surrogate_data[:, ranks] = sorted_data

    end_time = time.process_time()
    logger.info("Time to generate surrogates: %s", str(end_time - start_time))

    return surrogate_data


class ResultReconstructionData:
    """Creates a data object from file and or function definitions for use in
    array creation methods.

    """

    def __init__(self, mode: RunModes, case: str):
        # Get locations from configuration file
        (
            self.sav_loc,
            self.case_config_loc,
            self.case_dir,
            _,
        ) = config_setup.run_setup(mode, case)
        # Load case config file
        with open(
            Path(self.case_config_loc, "resultreconstruction.json"),
            encoding="utf-8",
        ) as f:
            self.case_config = json.load(f)

        # Get data type
        self.datatype = self.case_config["datatype"]

        self.case = case

        # Get scenarios
        self.scenarios = self.case_config["scenarios"]
        self.case = case
        self.bias_correction = False
        # Make it False for now, might change this default in future
        self.mi_scale = False

    def setup_scenario(self, scenario: str):
        """Retrieves data particular to each scenario for the case
        being investigated."""

        scenario_config = self.case_config[scenario]

        if scenario_config:
            if self.datatype == "file":
                self.bias_correction = scenario_config.get("bias_correction", None)
                self.mi_scale = scenario_config.get("mi_scale", False)
        else:
            logger.info("Defaulting to no bias correction")


def process_aux_file(filename, bias_correct=True, mi_scale=False, allow_neg=False):
    """Processes an auxiliary file and returns a list of affected_vars,
    weight_array as well as relative significance weight array.

    Parameters:
        filename (string): path to auxiliary to process
        allow_neg (bool): if true, allows negative values in final weight arrays,
         otherwise sets them to zero.
        bias_correct (bool): if true, subtracts the mean of the null distribution off
         the final value in weight array

    """

    destination_vars = []
    weights = []
    no_significance_test_weights = []
    significance_weights = []
    delays = []
    significance_thresholds = []
    with open(filename, encoding="utf-8") as aux_file:
        aux_file_reader = csv.reader(aux_file, delimiter=",")
        for row_index, row in enumerate(aux_file_reader):
            if row_index == 0:
                # Find the indices of important rows
                destination_var_index = row.index("destination_variable")

                if "max_ent" in row:
                    max_val_index = row.index("max_ent")
                else:
                    max_val_index = row.index("max_corr")

                if "bias_mean" in row:
                    bias_mean_index = row.index("bias_mean")
                else:
                    bias_mean_index = None

                if "threshold" in row:
                    thresh_index = row.index("threshold")
                else:
                    thresh_index = row.index("threshold_correlation")

                if "mi_fwd" in row:  # This should be available by default
                    mi_index = row.index("mi_fwd")
                else:
                    mi_index = None

                pass_threshold_index = row.index("significance_threshold_passed")
                pass_directionality_index = row.index("directionality_check_passed")
                max_delay_index = row.index("max_delay")

            if row_index > 0:
                destination_vars.append(row[destination_var_index])

                # Test if weight failed threshold or directionality test and write as
                # zero if true

                # In rare cases it might be desired to allow negative values
                # (e.g. correlation test)
                # TODO: Put the allow_neg parameter in a configuration file
                # NOTE: allow_neg also removes significance testing

                weight_candidate = float(row[max_val_index])

                if allow_neg:
                    no_significance_test_weight = weight_candidate
                    significance_test_weight = weight_candidate
                else:
                    if weight_candidate > 0.0:
                        # Attach to no significance test result
                        no_significance_test_weight = weight_candidate
                        if (
                            row[pass_threshold_index] == "False"
                            or row[pass_directionality_index] == "False"
                        ):
                            significance_test_weight = 0.0
                        else:
                            # pass_threshold is either None or True
                            significance_test_weight = weight_candidate
                    else:
                        significance_test_weight = 0.0
                        no_significance_test_weight = 0.0

                # If both bias correction and MI scaling is to be performed,
                # bias correction should happen first

                # Perform bias correction if required
                if (
                    bias_correct
                    and bias_mean_index
                    and (significance_test_weight > 0.0)
                ):
                    significance_test_weight = significance_test_weight - float(
                        row[bias_mean_index]
                    )
                    if significance_test_weight < 0:
                        raise ValueError("Negative weight after subtracting bias mean")

                if (mi_scale and mi_index) and (significance_test_weight > 0.0):
                    significance_test_weight = significance_test_weight / float(
                        row[mi_index]
                    )

                weights.append(significance_test_weight)
                no_significance_test_weights.append(no_significance_test_weight)
                delays.append(float(row[max_delay_index]))

                threshold = float(row[thresh_index])
                significance_thresholds.append(threshold)

                # Test if significance test passed before assigning weight
                if (
                    row[pass_threshold_index] == "True"
                    and row[pass_directionality_index] == "True"
                ):
                    # If the threshold is negative, take the absolute value
                    # TODO: Need to think the implications of this through
                    if threshold != 0:
                        significance_weight = float(row[max_val_index]) / abs(threshold)
                        if significance_weight > 0.0:
                            significance_weights.append(significance_weight)
                        else:
                            significance_weights.append(0.0)
                    else:
                        significance_weights.append(0.0)

                else:
                    significance_weights.append(0.0)

    return (
        destination_vars,
        weights,
        no_significance_test_weights,
        significance_weights,
        delays,
        significance_thresholds,
    )


def create_arrays(data_dir: Path, variables, bias_correct, mi_scale, generate_diffs):
    """
    data_dir is the location of the auxiliary data and weights folders for the
    specific case that is under investigation

    variables is the list of variables

    """

    absolute_weight_array_name = "weight_absolute_arrays"
    directional_weight_array_name = "weight_directional_arrays"
    neutral_weight_array_name = "weight_arrays"
    difabsoluteweightarray_name = "dif_weight_absolute_arrays"
    difdirectionalweightarray_name = "dif_weight_directional_arrays"
    difneutralweightarray_name = "dif_weight_arrays"
    absolutesigweightarray_name = "sigweight_absolute_arrays"
    directionalsigweightarray_name = "sigweight_directional_arrays"
    neutralsigweightarray_name = "sigweight_arrays"
    absolutedelayarray_name = "delay_absolute_arrays"
    directionaldelayarray_name = "delay_directional_arrays"
    neutraldelayarray_name = "delay_arrays"
    absolutesigthresholdarray_name = "sigthreshold_absolute_arrays"
    directionalsigthresholdarray_name = "sigthreshold_directional_arrays"
    neutralsigthresholdarray_name = "sigthreshold_arrays"

    directories = next(os.walk(data_dir))[1]

    test_strings = ["auxdata_absolute", "auxdata_directional", "auxdata"]

    for test_string in test_strings:
        if test_string in directories:
            if test_string == "auxdata_absolute":
                weight_array_name = absolute_weight_array_name
                difweightarray_name = difabsoluteweightarray_name
                sigweightarray_name = absolutesigweightarray_name
                delay_array_name = absolutedelayarray_name
                sigthresholdarray_name = absolutesigthresholdarray_name
            elif test_string == "auxdata_directional":
                weight_array_name = directional_weight_array_name
                difweightarray_name = difdirectionalweightarray_name
                sigweightarray_name = directionalsigweightarray_name
                delay_array_name = directionaldelayarray_name
                sigthresholdarray_name = directionalsigthresholdarray_name
            elif test_string == "auxdata":
                weight_array_name = neutral_weight_array_name
                difweightarray_name = difneutralweightarray_name
                sigweightarray_name = neutralsigweightarray_name
                delay_array_name = neutraldelayarray_name
                sigthresholdarray_name = neutralsigthresholdarray_name

            boxes = next(os.walk(Path(data_dir, test_string)))[1]
            for box in boxes:
                box_dir = Path(data_dir, test_string, box)
                # Get list of source variables
                source_var_filenames = next(os.walk(box_dir))[2]
                source_vars = []
                destination_var_array = []
                weight_array = []
                nosigtest_weight_array = []
                sigweight_array = []
                delay_array = []
                sigthreshold_array = []
                for source_var_file in source_var_filenames:
                    source_vars.append(str(source_var_file[:-4]))

                    # Open auxfile and return weight array as well as
                    # significance relative weight arrays

                    # TODO: Confirm whether correlation test absolutes
                    # correlations before sending to auxfile
                    # Otherwise, the allow null must be used much more wisely
                    (
                        destination_vars,
                        weights,
                        nosigtest_weights,
                        significance_weights,
                        delays,
                        significance_thresholds,
                    ) = process_aux_file(
                        Path(box_dir, source_var_file),
                        bias_correct=bias_correct,
                        mi_scale=mi_scale,
                    )

                    destination_var_array.append(destination_vars)
                    weight_array.append(weights)
                    nosigtest_weight_array.append(nosigtest_weights)
                    sigweight_array.append(significance_weights)
                    delay_array.append(delays)
                    sigthreshold_array.append(significance_thresholds)

                # Write the arrays to file
                # Create a base array based on the full set of variables
                # found in the typical WeightCalcData object

                # Initialize matrix with variables written
                # in first row and column
                weights_matrix = np.zeros(
                    (len(variables) + 1, len(variables) + 1)
                ).astype(object)

                weights_matrix[0, 0] = ""
                weights_matrix[0, 1:] = variables
                weights_matrix[1:, 0] = variables

                nosigtest_weights_matrix = np.copy(weights_matrix)
                sigweights_matrix = np.copy(weights_matrix)
                delay_matrix = np.copy(weights_matrix)
                sigthresh_matrix = np.copy(weights_matrix)

                # Write results to appropriate entries in array
                for source_var_index, source_var in enumerate(source_vars):
                    source_var_loc = variables.index(source_var)
                    for destination_var_index, destination_var in enumerate(
                        destination_var_array[source_var_index]
                    ):
                        destination_var_loc = variables.index(destination_var)

                        weights_matrix[destination_var_loc + 1, source_var_loc + 1] = (
                            weight_array[source_var_index][destination_var_index]
                        )
                        nosigtest_weights_matrix[
                            destination_var_loc + 1, source_var_loc + 1
                        ] = nosigtest_weight_array[source_var_index][
                            destination_var_index
                        ]
                        sigweights_matrix[
                            destination_var_loc + 1, source_var_loc + 1
                        ] = sigweight_array[source_var_index][destination_var_index]
                        delay_matrix[destination_var_loc + 1, source_var_loc + 1] = (
                            delay_array[source_var_index][destination_var_index]
                        )
                        sigthresh_matrix[
                            destination_var_loc + 1, source_var_loc + 1
                        ] = sigthreshold_array[source_var_index][destination_var_index]

                # Write to CSV files
                weight_array_dir = Path(data_dir, weight_array_name, box)
                config_setup.ensure_existence(weight_array_dir)
                weight_filename = Path(weight_array_dir, "weight_array.csv")
                np.savetxt(weight_filename, weights_matrix, delimiter=",", fmt="%s")

                delay_array_dir = Path(data_dir, delay_array_name, box)
                config_setup.ensure_existence(delay_array_dir)
                delay_filename = Path(delay_array_dir, "delay_array.csv")
                np.savetxt(delay_filename, delay_matrix, delimiter=",", fmt="%s")

                dir_parts = get_folders(data_dir)
                if "sigtested" in dir_parts:
                    dir_parts[dir_parts.index("sigtested")] = "nosigtest"
                    nosigtest_save_dir = dir_parts[0]
                    for path_part in dir_parts[1:]:
                        nosigtest_save_dir = Path(nosigtest_save_dir, path_part)

                    nosigtest_weight_array_dir = Path(
                        nosigtest_save_dir, weight_array_name, box
                    )
                    config_setup.ensure_existence(nosigtest_weight_array_dir)

                    nosigtest_delay_array_dir = Path(
                        nosigtest_save_dir, delay_array_name, box
                    )
                    config_setup.ensure_existence(nosigtest_delay_array_dir)

                    delay_filename = Path(nosigtest_delay_array_dir, "delay_array.csv")
                    np.savetxt(delay_filename, delay_matrix, delimiter=",", fmt="%s")

                    weight_filename = Path(
                        nosigtest_weight_array_dir, "weight_array.csv"
                    )
                    np.savetxt(
                        weight_filename,
                        nosigtest_weights_matrix,
                        delimiter=",",
                        fmt="%s",
                    )

                    sigweightarray_dir = Path(data_dir, sigweightarray_name, box)
                    config_setup.ensure_existence(sigweightarray_dir)
                    sigweightfilename = Path(sigweightarray_dir, "sigweight_array.csv")
                    np.savetxt(
                        sigweightfilename,
                        sigweights_matrix,
                        delimiter=",",
                        fmt="%s",
                    )

                    sigthresholdarray_dir = Path(data_dir, sigthresholdarray_name, box)
                    config_setup.ensure_existence(sigthresholdarray_dir)
                    sigthresholdfilename = Path(
                        sigthresholdarray_dir, "sigthreshold_array.csv"
                    )
                    np.savetxt(
                        sigthresholdfilename,
                        sigthresh_matrix,
                        delimiter=",",
                        fmt="%s",
                    )

            if generate_diffs:
                boxes = list(boxes)
                boxes.sort()

                for boxindex, box in enumerate(boxes):
                    difweights_matrix = np.zeros(
                        (len(variables) + 1, len(variables) + 1)
                    ).astype(object)

                    difweights_matrix[0, 0] = ""
                    difweights_matrix[0, 1:] = variables
                    difweights_matrix[1:, 0] = variables

                    if boxindex > 0:
                        base_weight_array_dir = Path(
                            data_dir, weight_array_name, boxes[boxindex - 1]
                        )  # Already one behind
                        base_weight_array_filename = Path(
                            base_weight_array_dir, "weight_array.csv"
                        )
                        final_weight_array_dir = Path(data_dir, weight_array_name, box)
                        final_weight_array_filename = Path(
                            final_weight_array_dir, "weight_array.csv"
                        )

                        with open(base_weight_array_filename, encoding="utf-8") as f:
                            num_cols = len(f.readline().split(","))
                            f.seek(0)
                            base_weight_matrix = np.genfromtxt(
                                f,
                                usecols=range(1, num_cols),
                                skip_header=1,
                                delimiter=",",
                            )

                        with open(final_weight_array_filename, encoding="utf-8") as f:
                            num_cols = len(f.readline().split(","))
                            f.seek(0)
                            final_weight_matrix = np.genfromtxt(
                                f,
                                usecols=range(1, num_cols),
                                skip_header=1,
                                delimiter=",",
                            )

                        # Calculate difference and save to file
                        # TODO: Investigate effect of taking absolute of differences
                        difweights_matrix[1:, 1:] = abs(final_weight_matrix) - abs(
                            base_weight_matrix
                        )

                    difweightarray_dir = Path(data_dir, difweightarray_name, box)
                    config_setup.ensure_existence(difweightarray_dir)
                    difweightfilename = Path(difweightarray_dir, "dif_weight_array.csv")
                    np.savetxt(
                        difweightfilename,
                        difweights_matrix,
                        delimiter=",",
                        fmt="%s",
                    )

                    if "sigtested" in get_folders(data_dir):
                        nosigtest_difweights_matrix = np.zeros(
                            (len(variables) + 1, len(variables) + 1)
                        ).astype(object)

                        nosigtest_difweights_matrix[0, 0] = ""
                        nosigtest_difweights_matrix[0, 1:] = variables
                        nosigtest_difweights_matrix[1:, 0] = variables

                        if boxindex > 0:
                            nosigtest_base_weight_array_dir = Path(
                                nosigtest_save_dir,
                                weight_array_name,
                                boxes[boxindex - 1],
                            )
                            nosigtest_base_weight_array_filename = Path(
                                nosigtest_base_weight_array_dir,
                                "weight_array.csv",
                            )
                            nosigtest_final_weight_array_dir = Path(
                                nosigtest_save_dir, weight_array_name, box
                            )
                            nosigtest_final_weight_array_filename = Path(
                                nosigtest_final_weight_array_dir,
                                "weight_array.csv",
                            )

                            with open(
                                nosigtest_base_weight_array_filename,
                                encoding="utf-8",
                            ) as f:
                                num_cols = len(f.readline().split(","))
                                f.seek(0)
                                nosigtest_base_weight_matrix = np.genfromtxt(
                                    f,
                                    usecols=range(1, num_cols),
                                    skip_header=1,
                                    delimiter=",",
                                )

                            with open(
                                nosigtest_final_weight_array_filename,
                                encoding="utf-8",
                            ) as f:
                                num_cols = len(f.readline().split(","))
                                f.seek(0)
                                nosigtest_final_weight_matrix = np.genfromtxt(
                                    f,
                                    usecols=range(1, num_cols),
                                    skip_header=1,
                                    delimiter=",",
                                )

                            # Calculate difference and save to file
                            # TODO: Investigate effect of taking absolute of differences
                            nosigtest_difweights_matrix[1:, 1:] = abs(
                                nosigtest_final_weight_matrix
                            ) - abs(nosigtest_base_weight_matrix)

                        nosigtest_difweightarray_dir = Path(
                            nosigtest_save_dir, difweightarray_name, box
                        )
                        config_setup.ensure_existence(nosigtest_difweightarray_dir)
                        nosigtest_difweightfilename = Path(
                            nosigtest_difweightarray_dir,
                            "dif_weight_array.csv",
                        )
                        np.savetxt(
                            nosigtest_difweightfilename,
                            nosigtest_difweights_matrix,
                            delimiter=",",
                            fmt="%s",
                        )

    return None


def create_signtested_directionalarrays(datadir, writeoutput):
    """Checks whether the directional weight arrays have corresponding
    absolute positive entries, writes another version with zeros if
    absolutes are negative.

    datadir is the location of the auxdata and weights folders for the
    specific case that is under investigation

    tsfilename is the file name of the original time series data file
    used to generate each case and is only used for generating a list of
    variables

    """

    signtested_weightarrayname = "signtested_weight_directional_arrays"
    signtested_sigweightarrayname = "signtested_sigweight_directional_arrays"

    directories = next(os.walk(datadir))[1]

    test_strings = [
        "weight_directional_arrays",
        "sigweight_directional_arrays",
    ]

    lookup_strings = ["weight_absolute_arrays", "sigweight_absolute_arrays"]

    boxfilenames = {
        "weight_absolute_arrays": "weight_array",
        "weight_directional_arrays": "weight_array",
        "sigweight_absolute_arrays": "sigweight_array",
        "sigweight_directional_arrays": "sigweight_array",
    }

    for test_index, test_string in enumerate(test_strings):
        if test_string in directories:
            if test_string == "weight_directional_arrays":
                signtested_directionalweightarrayname = signtested_weightarrayname
            if test_string == "sigweight_directional_arrays":
                signtested_directionalweightarrayname = signtested_sigweightarrayname

            boxes = next(os.walk(Path(datadir, test_string)))[1]
            for box in boxes:
                dirboxdir = Path(datadir, test_string, box)
                absboxdir = Path(datadir, lookup_strings[test_index], box)

                # Read the contents of the test_string array
                dir_arraydf = pd.read_csv(
                    Path(dirboxdir, boxfilenames[test_string] + ".csv")
                )
                # Read the contents of the comparative lookup_string array
                abs_arraydf = pd.read_csv(
                    Path(
                        absboxdir,
                        boxfilenames[lookup_strings[test_index]] + ".csv",
                    )
                )

                # Causevars is the first line of the array being read
                # Affectedvars is the first column of the array being read

                causevars = [
                    dir_arraydf.columns[1:][i]
                    for i in range(0, len(dir_arraydf.columns[1:]))
                ]

                affectedvars = [
                    dir_arraydf[dir_arraydf.columns[0]][i]
                    for i in range(0, len(dir_arraydf[dir_arraydf.columns[0]]))
                ]

                # Create directional signtested array
                signtested_dir_array = np.zeros(
                    (len(affectedvars) + 1, len(causevars) + 1)
                ).astype(object)

                # Initialize array with causevar labels in first column
                # and affectedvar labels in first row
                signtested_dir_array[0, 0] = ""
                signtested_dir_array[0, 1:] = causevars
                signtested_dir_array[1:, 0] = affectedvars

                for causevarindex in range(len(causevars)):
                    for affectedvarindex in range(len(affectedvars)):
                        # Check the sign of the abs_arraydf for this entry
                        abs_value = abs_arraydf[abs_arraydf.columns[causevarindex + 1]][
                            affectedvarindex
                        ]

                        if abs_value > 0:
                            signtested_dir_array[
                                affectedvarindex + 1, causevarindex + 1
                            ] = dir_arraydf[dir_arraydf.columns[causevarindex + 1]][
                                affectedvarindex
                            ]
                        else:
                            signtested_dir_array[
                                affectedvarindex + 1, causevarindex + 1
                            ] = 0

                # Write to CSV file
                if writeoutput:
                    signtested_weightarray_dir = Path(
                        datadir, signtested_directionalweightarrayname, box
                    )
                    config_setup.ensure_existence(signtested_weightarray_dir)

                    signtested_weightfilename = Path(
                        signtested_weightarray_dir,
                        boxfilenames[test_string] + ".csv",
                    )
                    np.savetxt(
                        signtested_weightfilename,
                        signtested_dir_array,
                        delimiter=",",
                        fmt="%s",
                    )

    return None


def extract_trends(datadir, writeoutput):
    """
    datadir is the location of the weight_array and delay_array folders for the
    specific case that is under investigation

    tsfilename is the file name of the original time series data file
    used to generate each case and is only used for generating a list of
    variables

    """

    # Create array to trend name dictionary
    namesdict = {
        "weight_absolute_arrays": "weight_absolute_trend",
        "weight_directional_arrays": "weight_directional_trend",
        "signtested_weight_directional_arrays": "signtested_weight_directional_trend",
        "weight_arrays": "weight_trend",
        "sigweight_absolute_arrays": "sigweight_absolute_trend",
        "sigweight_directional_arrays": "sigweight_directional_trend",
        "signtested_sigweight_directional_arrays": (
            "signtested_sigweight_directional_trend"
        ),
        "sigweight_arrays": "sigweight_trend",
        "delay_absolute_arrays": "delay_absolute_trend",
        "delay_directional_arrays": "delay_directional_trend",
        "delay_arrays": "delay_trend",
        "sigthreshold_absolute_arrays": "sigthreshold_absolute_trend",
        "sigthreshold_directional_arrays": "sigthreshold_directional_trend",
        "sigthreshold_arrays": "sigthreshold_trend",
    }

    boxfilenames = {
        "weight_absolute_arrays": "weight_array",
        "weight_directional_arrays": "weight_array",
        "signtested_weight_directional_arrays": "weight_array",
        "weight_arrays": "weight_array",
        "sigweight_absolute_arrays": "sigweight_array",
        "sigweight_directional_arrays": "sigweight_array",
        "signtested_sigweight_directional_arrays": "sigweight_array",
        "sigweight_arrays": "sigweight_array",
        "delay_absolute_arrays": "delay_array",
        "delay_directional_arrays": "delay_array",
        "delay_arrays": "delay_array",
        "sigthreshold_absolute_arrays": "sigthreshold_array",
        "sigthreshold_directional_arrays": "sigthreshold_array",
        "sigthreshold_arrays": "sigthreshold_array",
    }

    directories = next(os.walk(datadir))[1]

    test_strings = namesdict.keys()

    savedir = change_dirtype(datadir, "weight_data", "trends")

    for test_string in test_strings:
        if test_string in directories:
            trendname = namesdict[test_string]

            arraydataframes = []

            boxes = next(os.walk(Path(datadir, test_string)))[1]
            for box in boxes:
                boxdir = Path(datadir, test_string, box)

                # Read the contents of the test_string array
                arraydf = pd.read_csv(Path(boxdir, boxfilenames[test_string] + ".csv"))

                arraydataframes.append(arraydf)

            # Causevars is the first line of the array being read
            # Affectedvars is the first column of the array being read

            arraydf = arraydataframes[0]

            causevars = [
                arraydf.columns[1:][i] for i in range(0, len(arraydf.columns[1:]))
            ]
            affectedvars = [
                arraydf[arraydf.columns[0]][i]
                for i in range(0, len(arraydf[arraydf.columns[0]]))
            ]

            for causevar in causevars:
                # Create array with trends for specific causevar
                trend_array = np.zeros(
                    (len(affectedvars), len(boxes) + 1), dtype=object
                )
                # Initialize array with affectedvar labels in first row
                trend_array[:, 0] = affectedvars

                for affectedvarindex in range(len(affectedvars)):
                    trendvalues = []
                    for arraydf in arraydataframes:
                        trendvalues.append(arraydf[causevar][affectedvarindex])
                    trend_array[affectedvarindex:, 1:] = trendvalues

                # Write to CSV file
                if writeoutput:
                    trend_dir = Path(savedir, causevar)
                    config_setup.ensure_existence(trend_dir)

                    trendfilename = Path(trend_dir, trendname + ".csv")
                    np.savetxt(trendfilename, trend_array.T, delimiter=",", fmt="%s")

    return None


def result_reconstruction(mode: RunModes, case: str) -> None:
    """Reconstructs the weight_array and delay_array for different weight types
    from data generated by run_weightcalc process.

    WIP:
    For transient cases, generates difference arrays between boxes.

    The results are written to the same folders where the files are found.


    """

    result_reconstruction_data = ResultReconstructionData(mode, case)

    weight_calc_data = weightcalc.WeightCalcData(mode, case, False, False, False, False)

    saveloc, caseconfigdir, _, _ = config_setup.run_setup(mode, case)

    with open(Path(caseconfigdir, "weightcalc.json"), encoding="utf-8") as f:
        caseconfig = json.load(f)

    # Directory where subdirectories for scenarios will be stored
    scenariosdir = Path(saveloc, "weight_data", case)

    # Get list of all scenarios
    scenarios = next(os.walk(scenariosdir))[1]

    for scenario in scenarios:
        logger.info("Processing scenario: %s", scenario)

        result_reconstruction_data.setup_scenario(scenario)

        weight_calc_data.set_settings(scenario, caseconfig[scenario]["settings"][0])

        methodsdir = Path(scenariosdir, scenario)
        methods = next(os.walk(methodsdir))[1]
        for method in methods:
            logger.info("Processing method: %s", method)
            sigtypesdir = Path(methodsdir, method)
            sigtypes = next(os.walk(sigtypesdir))[1]
            for sigtype in sigtypes:
                logger.info("Processing sigtype: %s", sigtype)
                embedtypesdir = Path(sigtypesdir, sigtype)
                embedtypes = next(os.walk(embedtypesdir))[1]
                for embedtype in embedtypes:
                    logger.info("Processing embedtype: %s", embedtype)
                    datadir = Path(embedtypesdir, embedtype)
                    create_arrays(
                        datadir,
                        weight_calc_data.variables,
                        result_reconstruction_data.bias_correction,
                        result_reconstruction_data.mi_scale,
                        weight_calc_data.generate_diffs,
                    )
                    # Provide directional array version tested with absolute
                    # weight sign
                    # create_signtested_directionalarrays(datadir, writeoutput)

    return None


def trend_extraction(mode: str, case: str, write_output: bool) -> None:
    """Extracts dynamic trend of weights and delays out of weight_array and delay_array
    results between multiple boxes generated by the run_createarrays process for
    transient cases.

    The results are written to the trends results directory.

    """

    saveloc, _, _, _ = config_setup.run_setup(mode, case)

    # Directory where subdirectories for scenarios will be stored
    scenariosdir = Path(saveloc, "weight_data", case)

    # Get list of all scenarios
    scenarios = next(os.walk(scenariosdir))[1]

    for scenario in scenarios:
        logger.info("Extracting trends for scenario: %s", scenario)

        methodsdir = Path(scenariosdir, scenario)
        methods = next(os.walk(methodsdir))[1]
        for method in methods:
            logger.info("Processing method: %s", method)
            sig_types_dir = Path(methodsdir, method)
            sigtypes = next(os.walk(sig_types_dir))[1]
            for sig_type in sigtypes:
                logger.info("Processing sigtype: %s", sig_type)
                embed_types_dir = Path(sig_types_dir, sig_type)
                embedtypes = next(os.walk(embed_types_dir))[1]
                for embedtype in embedtypes:
                    logger.info("Processing embedtype: %s", embedtype)
                    datadir = Path(embed_types_dir, embedtype)
                    extract_trends(datadir, write_output)


def csv_to_h5(saveloc, raw_tsdata, scenario, case, overwrite=True):
    # Name the dataset according to the scenario
    dataset = scenario

    datapath = config_setup.ensure_existence(Path(saveloc, "data", case), make=True)

    filename = Path(datapath, scenario + ".h5")

    if overwrite or (not os.path.exists(filename)):
        hdf5writer = tb.open_file(filename, "w")
        data = np.genfromtxt(raw_tsdata, delimiter=",")
        # Strip time column and labels first row
        data = data[1:, 1:]
        array = hdf5writer.create_array(hdf5writer.root, dataset, data)

        array.flush()
        hdf5writer.close()

    return datapath


def read_timestamps(raw_tsdata: str | Path) -> list[str]:
    timestamps = []
    with open(raw_tsdata, encoding="utf-8") as f:
        datareader = csv.reader(f)
        for rowindex, row in enumerate(datareader):
            if rowindex > 0:
                timestamps.append(row[0])
    timestamps = np.asarray(timestamps)
    return timestamps


def read_variables(raw_tsdata: str | Path) -> list[str]:
    with open(raw_tsdata, encoding="utf-8") as f:
        variables = next(csv.reader(f))[1:]
    return variables


def writecsv(
    filename: str | Path, items: list | NDArray, header: list[str] | None = None
) -> None:
    """Write CSV directly"""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        if header is not None:
            csv.writer(f).writerow(header)
        csv.writer(f).writerows(items)


def change_dirtype(datadir, oldtype, newtype):
    dirparts = get_folders(datadir)
    dirparts[dirparts.index(oldtype)] = newtype
    datadir = dirparts[0]
    for pathpart in dirparts[1:]:
        datadir = Path(datadir, pathpart)

    return datadir


def fft_calculation(
    headerline,
    normalised_tsdata,
    variables,
    sampling_rate,
    sampling_unit,
    saveloc,
    case,
    scenario,
    plotting=False,
    plotting_endsample=500,
):
    # TODO: Perform detrending
    # log.info("Starting FFT calculations")
    # Using a print command instead as logging is late
    logger.info("Starting FFT calculations")

    # Change first entry of headerline from "Time" to "Frequency"
    headerline[0] = "Frequency"

    # Get frequency list (this is the same for all variables)
    freqlist = np.fft.rfftfreq(len(normalised_tsdata[:, 0]), sampling_rate)

    freqlist = freqlist[:, np.newaxis]

    fft_data = np.zeros((len(freqlist), len(variables)))

    def filename(name):
        return filename_template.format(case, scenario, name)

    for var_index, variable in enumerate(variables):
        var_data = normalised_tsdata[:, var_index]

        # Compute FFT (normalised amplitude)
        var_fft = abs(np.fft.rfft(var_data)) * (2.0 / len(var_data))

        fft_data[:, var_index] = var_fft

        if plotting:
            plt.figure()
            plt.plot(
                freqlist[0:plotting_endsample],
                var_fft[0:plotting_endsample],
                "r",
                label=variable,
            )
            plt.xlabel("Frequency (1/" + sampling_unit + ")")
            plt.ylabel("Normalised amplitude")
            plt.legend()

            plotdir = config_setup.ensure_existence(
                Path(saveloc, "fftplots"), make=True
            )

            filename_template = os.path.join(plotdir, "FFT_{}_{}_{}.pdf")

            plt.savefig(filename(variable))
            plt.close()

    #    varmaxindex = var_fft.tolist().index(max(var_fft))
    #    print variable + " maximum signal strenght frequency: " + \
    #        str(freqlist[varmaxindex])

    # Combine frequency list and FFT data
    datalines = np.concatenate((freqlist, fft_data), axis=1)

    # Define export directories and filenames
    datadir = config_setup.ensure_existence(Path(saveloc, "fftdata"), make=True)

    filename_template = os.path.join(datadir, "{}_{}_{}.csv")

    writecsv(filename("fft"), datalines, headerline)

    logger.info("Done with FFT calculations")

    return None


def write_box_dates(box_dates, save_loc, case, scenario):
    def filename(name):
        return filename_template.format(case, scenario, name)

    datadir = config_setup.ensure_existence(Path(save_loc, "boxdates"), make=True)
    filename_template = os.path.join(datadir, "{}_{}_{}.csv")

    headerline = ["Box index", "Box start", "Box end"]
    datalines = np.zeros((len(box_dates), 3))
    for index, box_date in enumerate(box_dates):
        box_index = index + 1
        box_start = box_date[0]
        box_end = box_date[-1]
        datalines[index, :] = [box_index, box_start, box_end]

    writecsv(filename("boxdates"), datalines, headerline)

    return None


def bandgap(min_freq, max_freq, vardata):
    """Bandgap filter based on FFT/IFFT concatenation"""
    # TODO: Add buffer values in order to prevent ringing
    freqlist = np.fft.rfftfreq(vardata.size, 1)
    # Investigate effect of using abs()
    var_fft = np.fft.rfft(vardata)
    cut_var_fft = var_fft.copy()
    cut_var_fft[(freqlist < min_freq)] = 0
    cut_var_fft[(freqlist > max_freq)] = 0

    cut_vardata = np.fft.irfft(cut_var_fft)

    return cut_vardata


def bandgapfilter_data(
    raw_tsdata,
    normalised_tsdata,
    variables,
    low_freq,
    high_freq,
    saveloc,
    case,
    scenario,
):
    """Bandgap filter data between the specified high and low frequenices.
    Also writes filtered data to standard format for easy analysis in
    other software, for example TOPCAT.

    """

    # TODO: add two buffer indices to the start and end to eliminate ringing
    # Header and time from main source file
    headerline = np.genfromtxt(raw_tsdata, delimiter=",", dtype="string")[0, :]
    time_values = np.genfromtxt(raw_tsdata, delimiter=",")[1:, 0]
    time_values = time_values[:, np.newaxis]

    # Compensate for the fact that there is one less entry returned if the
    # number of samples is odd
    if bool(normalised_tsdata.shape[0] % 2):
        inputdata_bandgapfiltered = np.zeros(
            (normalised_tsdata.shape[0] - 1, normalised_tsdata.shape[1])
        )
    else:
        inputdata_bandgapfiltered = np.zeros_like(normalised_tsdata)

    for varindex in range(len(variables)):
        vardata = normalised_tsdata[:, varindex]
        bandgapped_vardata = bandgap(low_freq, high_freq, vardata)
        inputdata_bandgapfiltered[:, varindex] = bandgapped_vardata

    if bool(normalised_tsdata.shape[0] % 2):
        # Only write from the second time entry as there is one less datapoint
        # TODO: Currently it seems to exclude the last instead? Confirm
        datalines = np.concatenate(
            (time_values[:-1], inputdata_bandgapfiltered), axis=1
        )
    else:
        datalines = np.concatenate((time_values, inputdata_bandgapfiltered), axis=1)

    # Define export directories and filenames
    datadir = config_setup.ensure_existence(Path(saveloc, "bandgappeddata"), make=True)

    filename_template = os.path.join(datadir, "{}_{}_{}_{}_{}.csv")

    def filename(name, lowfreq, highfreq):
        return filename_template.format(case, scenario, name, lowfreq, highfreq)

    # Store the normalised data in similar format as original data
    writecsv(
        filename("bandgapped_data", str(low_freq), str(high_freq)),
        datalines,
        headerline,
    )

    return inputdata_bandgapfiltered


def detrend_linear_model(data: NDArray) -> NDArray:
    df = pd.DataFrame(data)
    detrended_df = pd.DataFrame(signal.detrend(df.dropna(), axis=0))
    detrended_df.index = df.dropna().index
    detrended_df.columns = df.columns

    return detrended_df.dropna().values


def detrend_first_differences(data: NDArray) -> NDArray:
    df = pd.DataFrame(data)
    detrended_df = df - df.shift(1)
    # Make first entry zero
    detrended_df.iloc[0, :] = 0.0

    return detrended_df.dropna().values


def detrend_link_relatives(
    data: NDArray, cap_values: bool = True, cap: float = 20.0
) -> NDArray:
    # Multiply by 100 to get idea of percentage change in
    # Easier to make sense of when choosing estimator bandwidth, etc.
    # Subtract by unit to center around 0
    # TODO: Investigate effect of centering around 1 on single entropy estimates
    df = pd.DataFrame(data)
    detrended_df = ((df / df.shift(1)) - 1.0) * 100.0
    detrended_df.iloc[0, :] = 0.0

    # Cap at a specified difference
    if cap_values:
        detrended_df[detrended_df > cap] = cap
        detrended_df[detrended_df < -cap] = -cap

    return detrended_df.dropna().values


def skogestad_scale_select(
    vartype: str, lower_limit: float, nominal_level: float, high_limit: float
) -> float:
    if vartype == "D":
        limit = max((nominal_level - lower_limit), (high_limit - nominal_level))
    elif vartype == "S":
        limit = min((nominal_level - lower_limit), (high_limit - nominal_level))
    else:
        raise ValueError(f"Variable type flag not recognized: {vartype}")
    return limit


def skogestad_scale(data_raw, variables, scalingvalues):
    if scalingvalues is None:
        raise ValueError("Scaling values not defined")

    data_skogestadscaled = np.zeros_like(data_raw)

    scalingvalues["scale_factor"] = list(
        map(
            skogestad_scale_select,
            scalingvalues["vartype"],
            scalingvalues["low"],
            scalingvalues["nominal"],
            scalingvalues["high"],
        )
    )

    # Loop through variables
    # The variables are aligned with the columns in raw_data
    for index, var in enumerate(variables):
        factor = scalingvalues.loc[var]["scale_factor"]
        nominalval = scalingvalues.loc[var]["nominal"]
        data_skogestadscaled[:, index] = (data_raw[:, index] - nominalval) / factor

    return data_skogestadscaled


def write_normdata(saveloc, case, scenario, headerline, datalines):
    # Define export directories and filenames
    datadir = config_setup.ensure_existence(Path(saveloc, "normdata"), make=True)

    filename_template = os.path.join(datadir, "{}_{}_{}.csv")

    def filename(name):
        return filename_template.format(case, scenario, name)

    # Store the normalised data in similar format as original data
    writecsv(filename("normalised_data"), datalines, headerline)

    return None


def write_detrenddata(saveloc, case, scenario, headerline, datalines):
    # Define export directories and filenames
    datadir = config_setup.ensure_existence(Path(saveloc, "detrenddata"), make=True)

    filename_template = os.path.join(datadir, "{}_{}_{}.csv")

    def filename(name):
        return filename_template.format(case, scenario, name)

    # Store the detrended data in similar format as original data
    writecsv(filename("detrended_data"), datalines, headerline)

    return None


def normalise_data(
    headerline: list[str],
    timestamps: NDArray,
    inputdata_raw: NDArray,
    variables: list[str],
    saveloc: str | Path,
    case: str,
    scenario: str,
    method: str | bool,
    scalingvalues: dict | None,
) -> NDArray:
    if method == "standardise":
        inputdata_normalised = sklearn.preprocessing.scale(inputdata_raw, axis=0)
    elif method == "skogestad":
        inputdata_normalised = skogestad_scale(inputdata_raw, variables, scalingvalues)
    elif not method:
        # This also breaks when using linked relatives
        # Disable completely until full list of exlcusions properly implemented
        # # If method is simply false
        # # Still mean center the data
        # # This breaks when trying to use discrete methods
        # if 'transfer_entropy_discrete' not in weight_methods:
        #     inputdata_normalised = subtract_mean(inputdata_raw)
        # else:
        #     inputdata_normalised = inputdata_raw
        inputdata_normalised = inputdata_raw
    else:
        raise ValueError(f"Normalisation method not recognized: {method}")

    datalines = np.concatenate(
        (timestamps[:, np.newaxis], inputdata_normalised), axis=1
    )

    write_normdata(saveloc, case, scenario, headerline, datalines)

    return inputdata_normalised


def detrend_data(headerline, timestamps, inputdata, saveloc, case, scenario, method):
    if method == "first_differences":
        inputdata_detrended = detrend_first_differences(inputdata)
    elif method == "link_relatives":
        inputdata_detrended = detrend_link_relatives(inputdata)
    elif method == "linear_model":
        inputdata_detrended = detrend_linear_model(inputdata)
    elif not method:
        # If method is False
        # Write received data without any modifications
        inputdata_detrended = inputdata

    else:
        raise ValueError(f"Detrending method not recognized: {method}")

    datalines = np.concatenate((timestamps[:, np.newaxis], inputdata_detrended), axis=1)

    write_detrenddata(saveloc, case, scenario, headerline, datalines)

    return inputdata_detrended


def subtract_mean(inputdata_raw: NDArray) -> NDArray:
    """Subtracts mean from input data."""

    inputdata_lessmean = np.zeros_like(inputdata_raw)

    for col in range(inputdata_raw.shape[1]):
        colmean = np.mean(inputdata_raw[:, col])
        inputdata_lessmean[:, col] = inputdata_raw[:, col] - colmean
    return inputdata_lessmean


def read_connectionmatrix(connection_loc: str | Path) -> tuple[NDArray, list[str]]:
    """Imports the connection scheme for the data.
    The format of the CSV file should be:
    empty space, var1, var2, etc... (first row)
    var1, value, value, value, etc... (second row)
    var2, value, value, value, etc... (third row)
    etc...

    value = 1 if column variable points to row variable (causal relationship)
    value = 0 otherwise

    """
    with open(connection_loc, encoding="utf-8") as f:
        variables = next(csv.reader(f))[1:]
        connectionmatrix = np.genfromtxt(f, delimiter=",")[:, 1:]

    return connectionmatrix, variables


def read_scale_limits(scaling_loc: Path):
    """Imports the scale limits for the data.
    The format of the CSV file should be:
    var, low, nominal, high, vartype (first row)
    var1, float, float, float, ['D', 'S'] (second row)
    var2, float, float, float, ['D, 'S'] (third row)
    etc...

    type 'D' indicates disturbance variable and maximum deviation will be used
    type 'S' indicates state variable and minimum deviation will be used

    """
    scalingdf = pd.read_csv(scaling_loc)
    scalingdf.set_index("var", inplace=True)

    return scalingdf


def read_biasvector(biasvector_loc):
    """Imports the bias vector for faultmap purposes.
    The format of the CSV file should be:
    var1, var2, etc ... (first row)
    bias1, bias2, etc ... (second row)
    """
    with open(biasvector_loc, encoding="utf-8") as f:
        variables = next(csv.reader(f))[1:]
        biasvector = np.genfromtxt(f, delimiter=",")[:]
    return biasvector, variables


def read_header_values_datafile(location: str | Path) -> tuple[NDArray, list[str]]:
    """This method reads a CSV data file of the form:
    header, header, header, etc... (first row)
    value, value, value, etc... (second row)
    etc...

    """

    with open(location, encoding="utf-8") as f:
        header = next(csv.reader(f))[:]
        values = np.genfromtxt(f, delimiter=",")

    return values, header


def read_matrix(matrix_loc: str | Path) -> NDArray:
    """This method reads a matrix scheme for a specific scenario.

    Might need to pad matrix with zeros if it is non-square
    """
    with open(matrix_loc, encoding="utf-8") as f:
        matrix = np.genfromtxt(f, delimiter=",")

    # Remove labels
    matrix = matrix[1:, 1:]

    return matrix


def buildcase(dummyweight, digraph, name, dummycreation):
    if dummycreation:
        counter = 1
        for node in digraph.nodes():
            if digraph.in_degree(node) == 1:
                # TODO: Investigate the effect of different weights
                nameofscale = name + str(counter)
                digraph.add_edge(nameofscale, node, weight=dummyweight)
                digraph.add_node(nameofscale, bias=1.0)
                counter += 1

    connection = nx.to_numpy_array(digraph, weight=None)
    gain = nx.to_numpy_array(digraph, weight="weight")
    variablelist = digraph.nodes()
    nodedatalist = digraph.nodes(data=True)

    biaslist = []
    for node in digraph.nodes():
        biaslist.append(nodedatalist[node]["bias"])

    return np.array(connection), gain, variablelist, biaslist


def build_graph(
    variables: list[str],
    gain_matrix: NDArray,
    connections: NDArray,
    bias_vector: NDArray,
) -> nx.DiGraph:
    """
    Builds a directed graph using the given variables, gain matrix,
    connections, and bias vector.

    Args:
        variables (list): A list of variable names.
        gain_matrix (numpy.ndarray): A 2D array of gains.
        connections (numpy.ndarray): A 2D array of connections.
        bias_vector (numpy.ndarray): A 1D array of biases.

    Returns:
        networkx.DiGraph: A directed graph with weights and biases.
    """
    digraph = nx.DiGraph()
    # Construct the graph with connections
    for col, colvar in enumerate(variables):
        for row, rowvar in enumerate(variables):
            # The node order is source, sink according to
            # the convention that columns are sources and rows are sinks
            if connections[row, col] != 0:
                digraph.add_edge(rowvar, colvar, weight=gain_matrix[row, col])

    # Add the bias information to the graph nodes
    for nodeindex, nodename in enumerate(variables):
        digraph.add_node(nodename, bias=bias_vector[nodeindex])

    return digraph


def write_dictionary(filename, dictionary):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dictionary, f)


def rank_backward(
    variables, gainmatrix, connections, biasvector, dummyweight, dummycreation
):
    """This method adds a unit gain node to all nodes with an out-degree of 1 in order
    for the relative scale to be retained. Therefore all nodes with pointers should have
    2 or more edges pointing away from them.

    It uses the number of dummy variables to construct these gain, connection and
    variable name matrices.

    """

    # TODO: Modify bias vector to assign zero weight to all dummy nodes

    digraph = build_graph(variables, gainmatrix, connections, biasvector)
    return buildcase(dummyweight, digraph, "DV BWD ", dummycreation)


def get_box_endates(clean_df, window, overlap, freq):
    """Gets the end dates of boxes from dataframe that are continous over window and
    guarenteed to have a maximum overlap.

    clean_df: clean dataframe with nan assigned to all bad data
    window: size of window in steps at desired frequency
    overlap: size of minimum overlap desired in steps at desired frequency
    """

    # Calculate the minimum timedelta between start of boxes
    min_timedelta = pd.Timedelta(freq) * overlap

    clean_df = clean_df.resample(freq).mean()  # Resamples at desired frequency

    # Any aggregate function that returns nan when a nan occurs in window is suitable
    rolling_clean_df = clean_df.rolling(window=window, min_periods=window).mean()
    rolling_clean_df.dropna(
        inplace=True
    )  # All indexes that remain have window continous samples at freq

    end_indexes = [rolling_clean_df.index[0]]  # Initialise with first index
    next_box_index = 0
    next_box_exists = True
    gc.disable()
    while next_box_exists:
        logger.info("Bins identified: %s", str(len(end_indexes)))
        # Get current list of differences
        index_diffs = rolling_clean_df.index - rolling_clean_df.index[next_box_index]
        # Get index of first entry that is within outside the minimum overlap range
        try:
            next_box_index = next(
                index
                for index, timedelta in enumerate(index_diffs)
                if timedelta >= pd.Timedelta(min_timedelta)
            )
            # Append this end_index
            end_indexes.append(rolling_clean_df.index[next_box_index])
        except StopIteration:
            next_box_exists = False
    gc.enable()

    return end_indexes


def get_continuous_boxes(clean_df, window, overlap, freq):
    """
    Splits a DataFrame into continuous boxes of a specified window
    size and overlap.

    Args:
        clean_df (pandas.DataFrame): The DataFrame to split.
        window (int): Window size in number of time steps.
        overlap (float): Overlap between windows as a fraction.
        freq (str): Frequency of the time series, e.g. '1H'.

    Returns:
        tuple: (array_boxes, boxdates) where array_boxes is a list
            of arrays per box and boxdates is a list of arrays with
            start/end timestamps per box.
    """
    box_end_dates = get_box_endates(clean_df, window, overlap, freq)
    boxdates = [
        np.asarray(
            [
                (box_end_date - (pd.Timedelta(freq) * window)).value // 10**9,
                box_end_date.value // 10**9,
            ]
        )
        for box_end_date in box_end_dates
    ]

    boxes = [
        clean_df[(box_end_date - (pd.Timedelta(freq) * (window - 1))) : box_end_date]
        for box_end_date in box_end_dates
    ]

    array_boxes = [np.asarray(box) for box in boxes]

    return array_boxes, boxdates


def split_time_series_data(
    input_data: NDArray, sample_rate: float, box_size: int, box_num: int
) -> list[NDArray]:
    """
    Splits the input data into arrays useful for analyzing the
    change of weights over time.

    Args:
        input_data (numpy.ndarray): A numpy array containing values for a single
            variable after sub-sampling.
        sample_rate (float): The rate of sampling in time units (after sub-sampling).
        box_size (int): The size of each returned dataset in time units.
        box_num (int): The number of boxes that need to be analyzed.

    Returns:
        list: A list of numpy arrays, where each array represents a box of data.

    Notes:
        Boxes are evenly distributed over the provided dataset. The boxes will overlap
        if box_size * box_num is more than the simulated time, and will have spaces
        between them if it is less.
    """
    # Get total number of samples
    samples = len(input_data)
    # Convert boxsize to number of samples
    box_size_samples = int(round(box_size / sample_rate))
    # Calculate starting index for each box

    if box_num == 1:
        boxes = [input_data]

    else:
        box_start_index = np.empty((1, box_num))[0]
        box_start_index[:] = np.nan
        box_start_index[0] = 0
        box_start_index[-1] = samples - box_size_samples
        samples_between = (float(samples - box_size_samples)) / float(box_num - 1)
        box_start_index[1:-1] = [
            round(samples_between * index) for index in range(1, box_num - 1)
        ]
        boxes = [
            input_data[
                int(box_start_index[i]) : int(box_start_index[i])
                + int(box_size_samples)
            ]
            for i in range(int(box_num))
        ]

    return boxes


def calc_signal_entropy(
    var_data, weight_calc_data: WeightCalcData, estimator: EntropyMethods = "kernel"
) -> float:
    """Calculates single signal differential entropies
    by making use of the JIDT continuous box-kernel implementation.

    """

    # Setup Java class for infodynamics toolkit
    entropy_calculator = infodynamics.setup_entropy_calculator(
        weight_calc_data.infodynamics_loc
    )

    entropy = infodynamics.calc_entropy(entropy_calculator, var_data, estimator)
    return entropy


def vectorselection(
    data: NDArray,
    timelag: int,
    sub_samples: int,
    k: int = 1,
    l: int = 1,  # noqa: E741
) -> tuple[NDArray, NDArray, NDArray]:
    """Generates sets of vectors from tags time series data
    for calculating transfer entropy.

    For notation references see Shu2013.

    Takes into account the time lag (number of samples between vectors of the
    same variable).

    In this application the prediction horizon (h) is set to equal
    to the time lag.

    The first vector in the data array should be the samples of the variable
    to be predicted (x) while the second vector should be sampled of the vector
    used to make the prediction (y).

    sub_samples is the amount of samples in the dataset used to calculate the
    transfer entropy between two vectors and must satisfy
    sub_samples <= samples

    The required number of samples is extracted from the end of the vector.
    If the vector is longer than the number of samples specified plus the
    desired time lag then the remained of the data will be discarded.

    k refers to the dimension of the historical data to be predicted (x)

    l refers to the dimension of the historical data used
    to do the prediction (y)

    """
    _, sample_n = data.shape
    x_pred = data[0, sample_n - sub_samples :]
    x_pred = x_pred[np.newaxis, :]

    x_hist = np.zeros((k, sub_samples))
    y_hist = np.zeros((l, sub_samples))

    for n in range(1, k + 1):
        # Original form according to Bauer (2007)
        # TODO: Provide for comparison
        # Modified form according to Shu & Zhao (2013)
        start_index = (sample_n - sub_samples) - timelag * (n - 1) - 1
        end_index = sample_n - timelag * (n - 1) - 1
        x_hist[n - 1, :] = data[1, start_index:end_index]
    for m in range(1, l + 1):
        start_index = (sample_n - sub_samples) - timelag * m - 1
        end_index = sample_n - timelag * m - 1
        y_hist[m - 1 :, :] = data[0, start_index:end_index]

    return x_pred, x_hist, y_hist
