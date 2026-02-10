"""Performs all analysis steps for all cases specified in the
configuration file.

"""

# Standard modules
import json
import logging
import multiprocessing
from pathlib import Path

from faultmap import config_setup
from faultmap.data_processing import result_reconstruction, trend_extraction
from faultmap.graphreduce import reduce_graph_scenarios
from faultmap.noderank import noderankcalc
from faultmap.type_definitions import RunModes
from faultmap.weightcalc import weight_calc
from plotting.plotter import draw_plot

logger = logging.getLogger(__name__)
logging.basicConfig(level="DEBUG")

# TODO: Move to class object
# TODO: Perform analysis on scenario level inside class object


def run_weight_calc(
    config_loc: Path, write_output: bool, mode: RunModes, case: str, robust: bool
) -> None:
    with open(Path(config_loc, "config_weightcalc.json"), encoding="utf-8") as f:
        weight_calc_config = json.load(f)

    # Flag indicating whether single signal entropy values for each
    # signal involved should be calculated
    single_entropies = weight_calc_config["calc_single_entropies"]
    # Flag indicating whether
    fft_calc = weight_calc_config["fft_calc"]
    do_multiprocessing = weight_calc_config["multiprocessing"]

    if robust:
        try:
            weight_calc(
                mode,
                case,
                write_output,
                single_entropies,
                fft_calc,
                do_multiprocessing,
            )
        except Exception as exc:
            raise RuntimeError(f"Weight calculation failed for case: {case}") from exc
    else:
        weight_calc(
            mode,
            case,
            write_output,
            single_entropies,
            fft_calc,
            do_multiprocessing,
        )


def run_createarrays(mode: RunModes, case: str, robust: bool) -> None:
    if robust:
        try:
            # Needs to execute twice for nosigtest cases if derived from
            # sigtest cases
            # TODO: Remove this requirement
            result_reconstruction(mode, case)
            result_reconstruction(mode, case)
        except Exception as exc:
            raise RuntimeError(f"Array creation failed for case: {case}") from exc
    else:
        result_reconstruction(mode, case)
        result_reconstruction(mode, case)


def run_trend_extraction(mode, case, robust, write_output):
    """Entry point for trend extraction."""
    if robust:
        try:
            trend_extraction(mode, case, write_output)
        except Exception as exc:
            raise RuntimeError(f"Trend extraction failed for case: {case}") from exc
    else:
        trend_extraction(mode, case, write_output)

    return None


def run_noderank(mode, case, robust, write_output):
    if robust:
        try:
            noderankcalc(mode, case, write_output)
        except Exception as exc:
            raise RuntimeError(f"Node faultmap failed for case: {case}") from exc
    else:
        noderankcalc(mode, case, write_output)

    return None


def run_graphreduce(mode, case, robust, write_output):
    if robust:
        try:
            reduce_graph_scenarios(mode, case, write_output)
        except Exception as exc:
            raise RuntimeError(f"Graph reduction failed for case: {case}") from exc
    else:
        reduce_graph_scenarios(mode, case, write_output)


def run_plotting(mode, case, robust, write_output):
    if robust:
        try:
            draw_plot(mode, case, write_output)
        except Exception as exc:
            raise RuntimeError(f"Plotting failed for case: {case}") from exc
    else:
        draw_plot(mode, case, write_output)

    return None


def run_all(mode: RunModes, robust=False):
    """Runs all cases or tests"""
    _, config_loc, _, _ = config_setup.get_locations(mode)
    with open(Path(config_loc, "config_full.json"), encoding="utf-8") as f:
        full_run_config = json.load(f)

    # Flag indicating whether calculated results should be written to disk
    write_output = full_run_config["write_output"]
    # Provide the mode and case names to calculate
    mode = full_run_config["mode"]
    cases = full_run_config["cases"]

    for case in cases:
        logger.info("Now attempting case: %s", case)
        run_weight_calc(config_loc, write_output, mode, case, robust)
        run_createarrays(mode, case, robust)
        run_trend_extraction(mode, case, robust, write_output)
        run_noderank(mode, case, robust, write_output)
        run_graphreduce(mode, case, robust, write_output)
        run_plotting(mode, case, robust, write_output)
        logger.info("Done with case: %s", case)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    run_all("cases")
