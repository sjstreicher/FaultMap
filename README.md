# FaultMap

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2543739.svg)](https://doi.org/10.5281/zenodo.2543739)
[![Documentation Status](https://readthedocs.org/projects/faultmap/badge/?version=latest)](https://faultmap.readthedocs.io/en/latest/?badge=latest)

## Introduction

FaultMap is a data-driven, model-free process fault detection and diagnosis tool.
Causal links between processes elements are identified using information theory measures (transfer entropy).
These links are then used to create a visual representation of the main flows of information (disturbances, etc.) among the process elements as a directed graph.
These directed graphs are useful as troubleshooting aids.

Network centrality algorithms are applied to determine the most influential elements based on the strength and quality of their influence on other connected nodes (eigenvector centrality).

For detailed usage guides, method descriptions, and API reference, see the [documentation on Read the Docs](https://faultmap.readthedocs.io/) or build locally with `sphinx-build -b html docs/ docs/_build/html`.

## Prerequisites

Most of the prerequisites are related to getting JPype to work correctly:

- Python 3.10+ with compatible C++ compiler
  - On Windows, compiling packages usually requires the `VC++ 2015.3 v14.00 (v140) toolset for desktop` to be installed from the Visual Studio installer.
- Java JDK 1.8.201+ (or latest Java 8 SDK)
  - The JAVA_HOME environment variable should point to the installation directory.

## Installation

```bash
git clone https://github.com/SimonStreicher/FaultMap.git
cd FaultMap
conda create --name faultmap python=3.10
conda activate faultmap
pip install -e .
pytest
```

A Docker image is available with all necessary packages and dependencies installed.

```bash
docker pull simonstreicher/faultmap
```

If you want to build locally, the Dockerfile can be found at [FaultMapDocker GitHub](https://github.com/SimonStreicher/FaultMapDocker).

## Setup

Create directories for storing the data, configuration files as well as results.
Create a file `case_config.json` in the root directory, similar to `test_config.json` which comes with the distribution (located in the `tests` directory).
Enter the full path to the data, configuration and results directories as well as the `infodynamics.jar` you want to use for [Java Information Dynamics Toolkit (JIDT)](https://github.com/jlizier/jidt) (the tested version is included in the distribution).

Example `case_config.json` file:

```json
{
  "data_loc": "~/faultmap/faultmap_data",
  "config_loc": "~/repos/faultmapconfigs",
  "save_loc": "~/faultmap/faultmap_results",
  "infodynamics_loc": "~/repos/FaultMap/infodynamics.jar"
}
```

## Configuration

Refer to the `example_configs` directory in the distribution for the required format of configuration files in order to fully define cases and scenarios.
The following configuration files are needed to fully specify a specific case:

1. `weightcalc.json`
2. `noderank.json`
3. `graphreduce.json`
4. `plotting.json`

## Execution

In order to calculate a full set of results for a specific case, make sure this case name is included in the `config_full.json` file in the directory defined under `config_loc` in the `case_config.json` file.

Example `config_full.json` file (also included in `example_configs` directory):

```json
{
  "mode": "cases",
  "write_output": true,
  "cases": [
    "tennessee_eastman"
  ]
}
```
