"""This module provides methods for calculating the gains (weights) of edges connecting
variables in the directed graph.

Calculation of both Pearson's correlation and transfer entropy is supported.
Transfer entropy is calculated according to the global average of local entropy method.
All weights are optimized with respect to time shifts between the time series data
vectors (i.e. cross-correlated).

The delay giving the maximum weight is returned, together with the maximum weights.

All weights are tested for significance.
The Pearson's correlation weights are tested for significance according to
the parameters presented by Bauer2005.
The transfer entropy weights are tested for significance using a non-parametric
rank-order method using surrogate data generated according to the iterative amplitude
adjusted Fourier transform method (iAAFT).

"""

# Standard libraries
import csv
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from faultmap import config_setup, data_processing, datagen, weightcalc_onesource
from faultmap.type_definitions import RunModes
from faultmap.weightcalculators import (
    CorrelationWeightCalculator,
    TransferEntropyWeightCalculator,
    WeightCalculator,
)


class WeightCalcData:
    """Creates a data object from files or functions for use in
    weight calculation methods.

    """

    def __init__(
        self,
        mode: RunModes,
        case: str,
        single_entropies: bool,
        fft_calc: bool,
        do_multiprocessing: bool,
        use_gpu,
    ):
        """
        Parameters
        ----------
        mode : str
            Either 'test' or 'cases'. Tests data are generated dynamically and
            stored in specified folders. Case data are read from file and
            stored under organized headings in the save_loc directory specified
            in config.json.
        case : str
            The name of the case that is to be run. Points to dictionary in
            either test or case config files.
        single_entropies : bool
            Flags whether the entropies of single signals should be calculated.
        fft_calc : bool
            Indicates whether the FFT of all individual signals should be
            calculated.
        do_multiprocessing : bool
            Indicates whether the weight calculation operations should run in
            parallel processing mode where all available CPU cores
            are utilized.

        """
        # Get file locations from configuration file
        (
            self.save_loc,
            self.case_config_dir,
            self.case_dir,
            self.infodynamics_loc,
        ) = config_setup.run_setup(mode, case)
        # Load case config file
        with open(
            os.path.join(self.case_config_dir, "weightcalc.json"), encoding="utf-8"
        ) as f:
            self.case_config = json.load(f)
        # Get data type
        self.datatype = self.case_config["datatype"]
        # Get scenarios
        self.scenarios = self.case_config["scenarios"]
        # Get methods
        self.methods = self.case_config["methods"]

        self.do_multiprocessing = do_multiprocessing
        self.use_gpu = use_gpu

        self.case_name = case

        # Flag for calculating single signal entropies
        self.single_entropies = single_entropies
        # Flag for calculating FFT of all signals
        self.fft_calc = fft_calc

        self.settings_set = None
        self.connections_used = None

    def scenario_data(self, scenario_name: str):
        """Retrieves data particular to each scenario for the case being
        investigated.

        Parameters
        ----------
            scenario_name : str
                Name of scenario to retrieve data for. Should be defined in
                config file.

        """
        logging.info("The scenario name is: %s", scenario_name)

        self.settings_set = self.case_config[scenario_name]["settings"]

    def set_settings(self, scenario, settings_name):
        self.connections_used = self.case_config[settings_name].get(
            "use_connections", False
        )
        if "transient" in self.case_config[settings_name]:
            self.transient = self.case_config[settings_name]["transient"]
        else:
            self.transient = False
            logging.info("Defaulting to single time region analysis")
        self.transient_method = self.case_config[settings_name].get(
            "transient_method", "legacy"
        )

        if "normalise" in self.case_config[settings_name]:
            self.normalise = self.case_config[settings_name]["normalise"]
        else:
            self.normalise = False
            logging.info("Defaulting to no normalisation")
        if "detrend" in self.case_config[settings_name]:
            self.detrend = self.case_config[settings_name]["detrend"]
        else:
            self.detrend = False
            logging.info("Defaulting to no detrending")
        self.sigtest = self.case_config[settings_name]["sigtest"]
        if self.sigtest:
            # The transfer entropy threshold calculation method be either 'sixsigma' or
            # 'rankorder'
            self.threshold_method = self.case_config[settings_name]["threshold_method"]
            # The transfer entropy surrogate generation method must be either
            # 'iAAFT' or 'random_shuffle'
            self.surrogate_method = self.case_config[settings_name]["surrogate_method"]

            self.all_thresh = self.case_config[settings_name].get("all_thresh", False)

        # Get sampling rate and unit name
        self.sampling_rate = self.case_config[settings_name]["sampling_rate"]
        self.sampling_unit = self.case_config[settings_name]["sampling_unit"]
        # Get starting index
        self.start_index = self.case_config[settings_name].get("start_index", 0)

        # Get parameters for Kraskov method
        if "transfer_entropy_kraskov" in self.methods:
            self.additional_parameters = self.case_config[settings_name][
                "additional_parameters"
            ]

        # Get parameters for kernel method
        if "transfer_entropy_kernel" in self.methods:
            if "kernel_width" in self.case_config[settings_name]:
                self.kernel_width = self.case_config[settings_name]["kernel_width"]
            else:
                self.kernel_width = None

        if self.datatype == "file":
            # Get path to time series data input file in standard format
            # described in documentation under "Input data formats"
            raw_tsdata = os.path.join(
                self.case_dir, "data", self.case_config[scenario]["data"]
            )

            # Retrieve connection matrix
            if self.connections_used:
                # Get connection (adjacency) matrix
                connection_loc = os.path.join(
                    self.case_dir,
                    "connections",
                    self.case_config[scenario]["connections"],
                )
                (
                    self.connectionmatrix,
                    _,
                ) = data_processing.read_connectionmatrix(connection_loc)

            # Read data into Pandas dataframe
            raw_df = pd.read_csv(raw_tsdata)
            raw_df["Time"] = pd.to_datetime(raw_df["Time"], unit="s")
            raw_df.set_index("Time", inplace=True)

            self.variables = list(raw_df.keys())
            self.timestamps = np.asarray(raw_df.index.astype(np.int64) // 10**9)
            self.header_line = ["Time"] + [var for var in self.variables]

            self.input_data_raw = np.asarray(raw_df)

            # Convert times eries data in CSV file to H5 data format
            # datapath = data_processing.csv_to_h5(self.saveloc, raw_tsdata,
            #                                      scenario, self.casename)
            # Read variables from orignal CSV file
            # self.variables = data_processing.read_variables(raw_tsdata)
            # self.timestamps = data_processing.read_timestamps(raw_tsdata)
            # # Get inputdata from H5 table created
            # self.inputdata_raw = np.array(h5py.File(os.path.join(
            #     datapath, scenario + '.h5'), 'r')[scenario])
            # self.headerline = np.genfromtxt(raw_tsdata, delimiter=',',
            #                                 dtype='str')[0, :]

        elif self.datatype == "function":
            raw_tsdata_gen = self.case_config[scenario]["datagen"]
            if self.connections_used:
                connectionloc = self.case_config[scenario]["connections"]
                # Get the variables and connection matrix
                self.variables, self.connectionmatrix = getattr(
                    datagen, connectionloc
                )()
            # TODO: Store function arguments in scenario config file
            params = self.case_config[settings_name]["datagen_params"]
            # Get inputdata
            self.input_data_raw = getattr(datagen, raw_tsdata_gen)(params)
            self.input_data_raw = np.asarray(self.input_data_raw)

            self.timestamps = np.arange(
                0,
                len(self.input_data_raw[:, 0]) * self.sampling_rate,
                self.sampling_rate,
            )

            self.header_line = ["Time"] + list(self.variables)

        # Perform normalisation
        # Retrieve scaling limits from file
        if self.normalise == "skogestad":
            # Get scaling parameters
            if "scalelimits" in self.case_config[scenario]:
                scaling_loc = os.path.join(
                    self.case_dir,
                    "scalelimits",
                    self.case_config[scenario]["scalelimits"],
                )
                scaling_values = data_processing.read_scale_limits(scaling_loc)
            else:
                raise ValueError(
                    "Scale limits reference missing from configuration file"
                )
        else:
            scaling_values = None

        self.normalised_input_data = data_processing.normalise_data(
            self.header_line,
            self.timestamps,
            self.input_data_raw,
            self.variables,
            self.save_loc,
            self.case_name,
            scenario,
            self.normalise,
            scaling_values,
        )

        # Get delay type
        self.delay_type = self.case_config[settings_name].get(
            "delay_type", "datapoints"
        )
        # Get bias correction parameter
        if "bias_correct" in self.case_config[scenario]:
            self.bias_correct = self.case_config[scenario]["bias_correct"]
        else:
            self.bias_correct = False

        # Get size of sample vectors for test
        # Must be smaller than number of samples
        self.test_size = self.case_config[settings_name]["test_size"]

        # Get number of delays to test
        test_delays = self.case_config[scenario]["test_delays"]

        if "bidirectional_delays" in self.case_config[scenario].keys():
            self.bidirectional_delays = self.case_config[scenario][
                "bidirectional_delays"
            ]
        else:
            self.bidirectional_delays = False

        if self.bidirectional_delays is True:
            delay_range = range(-test_delays, test_delays + 1)
        else:
            delay_range = range(test_delays + 1)

        # Define intervals of delays
        if self.delay_type == "datapoints":
            self.delays = delay_range

        elif self.delay_type == "intervals":
            # Test delays at specified intervals
            self.delayinterval = self.case_config[settings_name]["delay_interval"]

            self.delays = [(val * self.delayinterval) for val in delay_range]

        self.source_var_indexes = self.case_config[scenario].get(
            "source_var_indexes", "all"
        )
        if self.source_var_indexes == "all":
            self.source_var_indexes = range(len(self.variables))

        self.destination_var_indexes = self.case_config[scenario].get(
            "destination_var_indexes", "all"
        )

        if self.destination_var_indexes == "all":
            self.destination_var_indexes = range(len(self.variables))

        if "bandgap_filtering" in self.case_config[scenario]:
            bandgap_filtering = self.case_config[scenario]["bandgap_filtering"]
        else:
            bandgap_filtering = False
        if bandgap_filtering:
            low_freq = self.case_config[scenario]["low_freq"]
            high_freq = self.case_config[scenario]["high_freq"]
            self.inputdata_bandgapfiltered = data_processing.bandgapfilter_data(
                raw_tsdata,
                self.normalised_input_data,
                self.variables,
                low_freq,
                high_freq,
                self.save_loc,
                self.case_name,
                scenario,
            )
            self.inputdata_originalrate = self.inputdata_bandgapfiltered
        else:
            self.inputdata_originalrate = self.normalised_input_data

        # Perform detrending
        # Detrending should be performed after normalisation and band gap filtering

        self.inputdata_originalrate = data_processing.detrend_data(
            self.header_line,
            self.timestamps,
            self.inputdata_originalrate,
            self.save_loc,
            self.case_name,
            scenario,
            self.detrend,
        )

        # Subsample data if required
        # Get sub_sampling interval
        self.sub_sampling_interval = self.case_config[settings_name][
            "sub_sampling_interval"
        ]
        # TODO: Use proper pandas.tseries.resample techniques
        # if it will really add any functionality
        # TODO: Investigate use of forward-backward Kalman filters
        self.inputdata = self.inputdata_originalrate[0 :: self.sub_sampling_interval]

        if self.transient:
            self.boxsize = self.case_config[settings_name]["boxsize"]
            if self.transient_method == "legacy":
                self.boxnum = self.case_config[settings_name]["boxnum"]
            elif self.transient_method == "robust":
                self.boxoverlap = self.case_config[settings_name]["boxoverlap"]

        else:
            self.boxnum = 1  # Only a single box will be used
            self.boxsize = self.inputdata.shape[0] * self.sampling_rate
            # This box should now return the same size
            # as the original data file - but it does not play a role at all
            # in the actual box determination for the case of boxnum = 1

        if not hasattr(self, "transient_method"):
            self.transient_method = None

        if self.transient_method == "legacy" or self.transient_method is None:
            # Get box start and end dates
            self.boxdates = data_processing.split_time_series_data(
                self.timestamps,
                self.sampling_rate * self.sub_sampling_interval,
                self.boxsize,
                self.boxnum,
            )
            data_processing.write_box_dates(
                self.boxdates, self.save_loc, self.case_name, scenario
            )

            # Generate boxes to use
            self.boxes = data_processing.split_time_series_data(
                self.inputdata,
                self.sampling_rate * self.sub_sampling_interval,
                self.boxsize,
                self.boxnum,
            )

        elif self.transient_method == "robust":
            # Get box start and end dates

            df = pd.DataFrame(self.inputdata)
            df.index = pd.to_datetime(self.timestamps, unit="s")
            df.columns = self.variables

            freq_string = str(self.sampling_rate) + "S"

            self.boxes, self.boxdates = data_processing.get_continuous_boxes(
                df, self.boxsize, self.boxoverlap, freq_string
            )

            data_processing.write_box_dates(
                self.boxdates, self.save_loc, self.case_name, scenario
            )

            self.boxnum = len(self.boxdates)

        # Select which of the boxes to evaluate
        if self.transient:
            if "boxindexes" in self.case_config[scenario]:
                if self.case_config[scenario]["boxindexes"] == "range":
                    self.boxindexes = range(
                        self.case_config[scenario]["boxindexes_start"],
                        self.case_config[scenario]["boxindexes_end"] + 1,
                    )
                else:
                    self.boxindexes = self.case_config[scenario]["boxindexes"]
            else:
                self.boxindexes = "all"
            if self.boxindexes == "all":
                self.boxindexes = range(self.boxnum)
        else:
            self.boxindexes = [0]

        if len(self.boxindexes) > 1:
            self.generate_diffs = True
        else:
            self.generate_diffs = False

        # Calculate delays in indexes as well as time units
        if self.delay_type == "datapoints":
            self.actual_delays = [
                (delay * self.sampling_rate * self.sub_sampling_interval)
                for delay in self.delays
            ]
            self.sample_delays = self.delays
        elif self.delay_type == "intervals":
            self.actual_delays = [
                int(round(delay / self.sampling_rate)) * self.sampling_rate
                for delay in self.delays
            ]
            self.sample_delays = [
                int(round(delay / self.sampling_rate)) for delay in self.delays
            ]

        # Create descriptive dictionary for later use
        # This will need to be approached slightly differently to allow for
        # different formats under the same "plant"
        #        self.descriptions = data_processing.descriptive_dictionary(
        #            os.path.join(self.casedir, 'data', 'tag_descriptions.csv'))

        # FFT the data and write back in format that can be analysed in
        # TOPCAT in a plane plot

        if self.fft_calc:
            data_processing.fft_calculation(
                self.header_line,
                self.inputdata_originalrate,
                self.variables,
                self.sampling_rate,
                self.sampling_unit,
                self.save_loc,
                self.case_name,
                scenario,
            )


def writecsv_weightcalc(filename, items, header):
    """CSV writer customized for use in weightcalc function."""

    with open(filename, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(header)
        csv.writer(file).writerows(items)


def calculate_weights(
    weight_calc_data: WeightCalcData, method: str, scenario: str, write_output: bool
):
    """Determines the maximum weight between two variables by searching through
    a specified set of delays.

    Args:
        weight_calc_data:
        method: Can be one of the following:
            'cross_correlation'
            'partial_correlation' -- does not support time delays
            'transfer_entropy_kernel'
            'transfer_entropy_kraskov'
        scenario:
        write_output:

    TODO: Fix partial correlation method to make use of time delays

    Returns:

    """

    weight_calculator: WeightCalculator

    if method == "cross_correlation":
        weight_calculator = CorrelationWeightCalculator(weight_calc_data)
    elif method == "transfer_entropy_kernel":
        weight_calculator = TransferEntropyWeightCalculator(weight_calc_data, "kernel")
    elif method == "transfer_entropy_kraskov":
        weight_calculator = TransferEntropyWeightCalculator(weight_calc_data, "kraskov")
    elif method == "transfer_entropy_discrete":
        weight_calculator = TransferEntropyWeightCalculator(
            weight_calc_data, "discrete"
        )
    else:
        raise ValueError("Method not recognized")

    if weight_calc_data.sigtest:
        significance_status = "significance_tested"
    else:
        significance_status = "no_significance_test"

    if method == "transfer_entropy_kraskov":
        if weight_calc_data.additional_parameters["auto_embed"]:
            embed_status = "auto-embedding"
        else:
            embed_status = "naive"
    else:
        embed_status = "naive"

    var_dims = len(weight_calc_data.variables)
    start_index = weight_calc_data.start_index
    size = weight_calc_data.test_size

    source_delete_list = []
    destination_delete_list = []
    for index in range(var_dims):
        if index not in weight_calc_data.source_var_indexes:
            source_delete_list.append(index)
            logging.info("Deleted column %s", str(index))
        if index not in weight_calc_data.destination_var_indexes:
            destination_delete_list.append(index)
            logging.info("Deleted row %s", str(index))

    if weight_calc_data.connections_used:
        new_connection_matrix = weight_calc_data.connectionmatrix
    else:
        new_connection_matrix = np.ones((var_dims, var_dims))
    # Substitute columns not used with zeros in connection matrix
    for source_delete_index in source_delete_list:
        new_connection_matrix[:, source_delete_index] = np.zeros(var_dims)
    # Substitute rows not used with zeros in connection matrix
    for destination_delete_index in destination_delete_list:
        new_connection_matrix[destination_delete_index, :] = np.zeros(var_dims)

    # Initiate header line for weight store file
    # Create "Delay" as header for first row
    header_line = ["Delay"]
    for destination_var_index in weight_calc_data.destination_var_indexes:
        destination_var_name = weight_calc_data.variables[destination_var_index]
        header_line.append(destination_var_name)

    def filename(weight_name: str, box_index: int, source_var: str) -> Path:
        """Define filename structure for CSV file containing weights between a specific
        source variable and all the subsequent destination variables.

        Args:
            weight_name (str): Name of the weight.
            box_index (int): Index of the box.
            source_var (str): Name of the source variable.

        Returns:
            Path: The filename of the CSV file containing weights between a specific
                source variable and all the subsequent destination variables.
        """
        box_string = f"box{box_index:03d}"

        return Path(
            config_setup.ensure_existence(
                os.path.join(weight_store_dir, weight_name, box_string), make=True
            ),
            f"{source_var}.csv",
        )

    # Store the weight calculation results in similar format as original data

    # Define weight_store_dir up to the method level
    weight_store_dir = config_setup.ensure_existence(
        Path(
            weight_calc_data.save_loc,
            "weight_data",
            weight_calc_data.case_name,
            scenario,
            method,
            significance_status,
            embed_status,
        ),
        make=True,
    )

    if weight_calc_data.single_entropies:
        # Initiate header_line for single signal entropy storage file
        signal_entropy_header_line = weight_calc_data.variables
        # Define filename structure for CSV file

        def signal_entropy_filename(name, box_index):
            return signal_entropy_filename_template.format(
                weight_calc_data.case_name, scenario, name, box_index
            )

        signal_entropy_dir = config_setup.ensure_existence(
            Path(weight_calc_data.save_loc, "signal_entropies"), make=True
        )

        signal_entropy_filename_template = os.path.join(
            signal_entropy_dir, "{}_{}_{}_box{:03d}.csv"
        )

    for box_index in weight_calc_data.boxindexes:
        box = weight_calc_data.boxes[box_index]

        # Calculate single signal entropies - do not worry about
        # delays, but still do it according to different boxes
        if weight_calc_data.single_entropies:
            # Calculate single signal entropies of all variables
            # and save output in similar format to
            # standard weight calculation results
            signal_entropies = []
            for var_index, _ in enumerate(weight_calc_data.variables):
                var_data = box[:, var_index][start_index : start_index + size]
                entropy = data_processing.calc_signal_entropy(
                    var_data, weight_calc_data
                )
                signal_entropies.append(entropy)

            # Write the signal entropies to file - one file for each box
            # Each file will only have one line as we are not
            # calculating for different delays as is done for the case of
            # variable pairs.

            # Need to add another axis to signalentlist in order to make
            # it a sequence so that it can work with writecsv_weightcalc
            signal_entropies_array: NDArray = np.asarray(signal_entropies)
            signal_entropies = signal_entropies_array[np.newaxis, :]

            writecsv_weightcalc(
                signal_entropy_filename("signal_entropy", box_index + 1),
                signal_entropies,
                signal_entropy_header_line,
            )

        # Start parallelising code here
        # Create one process for each source variable
        non_iter_args = [
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
        ]

        # Run the script that will handle multiprocessing
        weightcalc_onesource.run(non_iter_args, weight_calc_data.do_multiprocessing)


def weight_calc(
    mode: RunModes,
    case: str,
    writeoutput: bool = False,
    single_entropies: bool = False,
    calc_fft: bool = False,
    do_multiprocessing: bool = False,
    use_gpu: bool = False,
) -> None:
    """Reports the maximum weight as well as associated delay
    obtained by shifting the affected variable behind the causal variable a
    specified set of delays.

    Parameters
    ----------
        mode : str
            Either 'test' or 'cases'. Tests data are generated dynamically and
            stored in specified folders. Case data are read from file
            and stored under organized headings in the saveloc directory
            specified in config.json.
        case : str
            The name of the case that is to be run. Points to dictionary in
            either test or case config files.
        single_entropies : bool
            Flags whether the entropies of single signals should be calculated.
        calc_fft : bool
            Indicates whether the FFT of all individual signals should be
            calculated.
        do_multiprocessing : bool
            Indicates whether the weight calculation operations should run in
            parallel processing mode where all available CPU cores
            are utilized.

    Notes
    -----
        Supports calculating weights according to either correlation or
        transfer entropy metrics.

    """

    weight_calc_data = WeightCalcData(
        mode, case, single_entropies, calc_fft, do_multiprocessing, use_gpu
    )

    for scenario in weight_calc_data.scenarios:
        logging.info("Running scenario %s", scenario)
        # Update scenario-specific fields of WeightCalcData object
        weight_calc_data.scenario_data(scenario)
        for settings_name in weight_calc_data.settings_set:
            weight_calc_data.set_settings(scenario, settings_name)
            logging.info("Now running settings %s", settings_name)

            for method in weight_calc_data.methods:
                logging.info("Method: %s", method)

                start_time = time.process_time()
                calculate_weights(weight_calc_data, method, scenario, writeoutput)
                end_time = time.process_time()
                logging.info("Weight calc time: %s", end_time - start_time)


if __name__ == "__main__":
    multiprocessing.freeze_support()
