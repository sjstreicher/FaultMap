"""Setup functions used to read configuration files."""

import json
import os
from pathlib import Path
from typing import NamedTuple

from faultmap.type_definitions import RunModes


class Locations(NamedTuple):
    """Directories used for data, configuration, results, and JIDT."""

    data_loc: Path
    config_loc: Path
    save_loc: Path
    infodynamics_loc: Path


class CaseSetup(NamedTuple):
    """Directories resolved for a specific case run."""

    save_loc: Path
    case_config_dir: Path
    case_dir: Path
    infodynamics_loc: Path


def ensure_existence(location: Path, make=True) -> Path:
    """

    Args:
        location:
        make:

    Returns:

    """
    if not os.path.exists(location):
        if make:
            os.makedirs(location)
        else:
            raise FileNotFoundError(f"File does not exist: {location}")
    return Path(location)


def get_locations(mode: RunModes = "cases") -> Locations:
    """Gets all required directories related to the specified mode.

    TODO: Remove the need for this by using proper test fixtures

    Parameters
    ----------
        mode : string
            Either 'test' or 'cases'. Specifies whether the test or user
            configurable cases directories should be set.
            Test directories are read from test_config.json which is bundled
            with the code, while cases directories are read from
            case_config.json which must be created by the user.

    Returns
    -------
    Locations
        A named tuple containing ``data_loc``, ``config_loc``,
        ``save_loc``, and ``infodynamics_loc`` paths.

    """
    # Load directories config file
    if mode == "test":
        parent_dir = Path(__file__).parent
        tests_dir = Path(parent_dir, "../tests")
        with open(Path(tests_dir, "test_config.json"), encoding="utf-8") as file:
            dirs = json.load(file)
    elif mode == "cases":
        with open("../case_config.json", encoding="utf-8") as file:
            dirs = json.load(file)
    else:
        raise ValueError(f"Mode name not recognized: {mode}")

    # Get data and preferred export directories from
    # directories config file
    locations = [
        ensure_existence(os.path.expanduser(dirs[location]))
        for location in ["data_loc", "config_loc", "save_loc", "infodynamics_loc"]
    ]
    return Locations(*locations)


def run_setup(mode: RunModes, case: str) -> CaseSetup:
    """Gets all required directories from the case configuration file.

    Args:
        mode: Either 'test' or 'cases'. Specifies whether the test or user configurable
            cases directories should be set. Test directories are read from
            test_config.json which is bundled with the code, while cases directories are
            read from case_config.json which must be created by the user.
        case: The name of the case that is to be run. Points to dictionary in either
            test or case config files.

    Returns:
        CaseSetup named tuple containing ``save_loc``, ``case_config_dir``,
        ``case_dir``, and ``infodynamics_loc`` paths.

    """

    locations = get_locations(mode)

    # Define case data directory
    case_dir = ensure_existence(Path(locations.data_loc, mode, case), make=True)
    case_config_dir = Path(locations.config_loc, mode, case)

    return CaseSetup(
        save_loc=locations.save_loc,
        case_config_dir=case_config_dir,
        case_dir=case_dir,
        infodynamics_loc=locations.infodynamics_loc,
    )
