"""Type definitions used throughout the library."""

from typing import Literal

RunModes = Literal["tests", "cases"]
MutualInformationMethods = Literal["kernel", "kraskov", "discrete"]
EntropyMethods = Literal["gaussian", "kernel", "kozachenko"]
