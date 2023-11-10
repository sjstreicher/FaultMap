"""Setup functions used to read configuration files."""

import json
import os
from pathlib import Path
from typing import Tuple

from faultmap.type_definitions import RunModes


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
            raise IOError(f"File does not exists: {location}")
    return Path(location)


def get_locations(mode: RunModes = "cases") -> tuple[Path, Path, Path, Path]:
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
        data_loc : path
        config_loc : path
        save_loc : path
        infodynamics_loc : path

    """
    # Load directories config file
    if mode == "test":
        parent_dir = Path(__file__).parent
        tests_dir = Path(parent_dir, "../tests")
        with open(Path(tests_dir, "test_config.json"), encoding="utf-8") as file:
            dirs = json.load(file)
        file.close()
    elif mode == "cases":
        with open("../case_config.json", encoding="utf-8") as file:
            dirs = json.load(file)
        file.close()
    else:
        raise NameError("Mode name not recognized")

    # Get data and preferred export directories from
    # directories config file
    locations = [
        ensure_existence(os.path.expanduser(dirs[location]))
        for location in ["data_loc", "config_loc", "save_loc", "infodynamics_loc"]
    ]
    data_loc, config_loc, save_loc, infodynamics_loc = locations

    return data_loc, config_loc, save_loc, infodynamics_loc


def run_setup(mode: RunModes, case: str) -> Tuple[Path, Path, Path, Path]:
    """Gets all required directories from the case configuration file.

    Args:
        mode: Either 'test' or 'cases'. Specifies whether the test or user configurable
            cases directories should be set. Test directories are read from
            test_config.json which is bundled with the code, while cases directories are
            read from case_config.json which must be created by the user.
        case: The name of the case that is to be run. Points to dictionary in either
            test or case config files.

    Returns:
        save_loc:
        case_config_dir:
        case_dir:
        infodynamics_loc:

    """

    data_loc, config_loc, save_loc, infodynamics_loc = get_locations(mode)

    # Define case data directory
    case_dir = ensure_existence(Path(data_loc, mode, case), make=True)
    case_config_dir = Path(config_loc, mode, case)

    return save_loc, case_config_dir, case_dir, infodynamics_loc
