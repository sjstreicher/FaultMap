"""Generates interactive HTML reports for all configured cases."""

import json
import logging
import os

from faultmap import config_setup
from faultmap.report import generate_report

logging.basicConfig(level=logging.INFO)

_, configloc, _, _ = config_setup.get_locations()
report_config = json.load(open(os.path.join(configloc, "config_report.json")))

mode = report_config["mode"]
cases = report_config["cases"]

for case in cases:
    generate_report(mode, case)
