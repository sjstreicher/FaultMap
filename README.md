# FaultMap

[![Travis Build](https://travis-ci.org/SimonStreicher/FaultMap.png?branch=master "Travis Build Status")](https://travis-ci.org/SimonStreicher/FaultMap)

[![Code Coverage](https://coveralls.io/repos/github/SimonStreicher/FaultMap/badge.svg?branch=master "Code Coverage")](https://coveralls.io/github/SimonStreicher/FaultMap?branch=master)

[![Code Climate](https://codeclimate.com/github/SimonStreicher/FaultMap/badges/gpa.svg "Code Climate Grade")](https://codeclimate.com/github/SimonStreicher/FaultMap)

[![DOI](https://zenodo.org/badge/14229559.svg "DOI")](https://zenodo.org/badge/latestdoi/14229559)

## Introduction

FaultMap is a data-driven, model-free process fault detection and diagnosis tool.
Causal links between processes elements are identified using information theory measures (transfer entropy).
These links are then used to create a visual representation of the main flows of information (disturbances, etc.) among the process elements as a directed graph.
These directed graphs are useful as troubleshooting aids.

Network centrality algorithms are applied to determine the most influential elements based on the strength and quality of their influence on other connected nodes (eigenvector centrality).

Documentation and demonstrations still under development.

## Prerequisites

Most of the prerequisites are related to getting JPype to work correctly:

- Python 3.7+ with compatible C++ compiler
  - On Windows, compiling packages usually requires the `VC++ 2015.3 v14.00 (v140) toolset for desktop` to be installed from the Visual Studio installer.
- Java JDK 1.8.201+ (or latest Java 8 SDK)
  - The JAVA_HOME environment variable should point to the installation directory.

## Installation

```bash
git clone https://github.com/SimonStreicher/FaultMap.git
cd FaultMap
conda create --name faultmap python=3.7
source activate faultmap
pip install -r requirements.txt
pytest
```

A Docker image is available with all necessary packages and dependencies installed.

```bash
docker pull simonstreicher/faultmap
```

If you want to build locally, the Dockerfile can be found at [FaultMapDocker GitHub](https://github.com/SimonStreicher/FaultMapDocker).

## Setup

Create directories for storing the data, configuration files as well as results.
Create a file `caseconfig.json` in the root directory, similar to `testconfig.json` which comes with the distribution.
Enter the full path to the data, configuration and results directories as well as the `infodynamics.jar` you want to use for [Java Information Dynamics Toolkit (JIDT)](https://github.com/jlizier/jidt) (the tested version is included in the distribution).

Example `caseconfig.json` file (also included in `example_configs` directory):

```json
{
  "dataloc": "~/faultmap/faultmap_data",
  "configloc": "~/repos/faultmapconfigs",
  "saveloc": "~/faultmap/faultmap_results",
  "infodynamicsloc": "~/repos/FaultMap/infodynamics.jar"
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

In order to calculate a full set of results for a specific case, make sure this case name is included in the `config_full.json` file in the directory defined under `configloc` in the `caseconfig.json` file.

Example `config_full.json` file (also included in `example_configs` directory):

```json
{
  "mode": "cases",
  "writeoutput": true,
  "cases": [
    "tennessee_eastman"
  ]
}
```
