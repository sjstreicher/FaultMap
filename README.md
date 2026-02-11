# FaultMap

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2543739.svg)](https://doi.org/10.5281/zenodo.2543739)
[![Documentation Status](https://readthedocs.org/projects/faultmap/badge/?version=latest)](https://faultmap.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/SimonStreicher/FaultMap/actions/workflows/ci.yml/badge.svg)](https://github.com/SimonStreicher/FaultMap/actions/workflows/ci.yml)

## Introduction

FaultMap is a data-driven, model-free process fault detection and diagnosis tool.
It identifies causal links between process elements using information-theoretic measures (transfer entropy) and represents the resulting information flows as directed graphs.
Network centrality algorithms (eigenvector centrality) are then applied to rank process elements by the strength and quality of their influence on connected nodes, enabling root cause identification.

These directed graphs and rankings serve as troubleshooting aids for plant-wide fault diagnosis.

## How It Works

FaultMap follows a three-step process:

1. **Weight calculation** -- Transfer entropy is computed between all pairs of process signals to quantify directional information flow. This produces edge weights for a directed graph.
2. **Node ranking** -- Graph centrality algorithms (primarily eigenvector centrality) are applied to the weighted directed graph to rank nodes by their influence on the network.
3. **Graph reduction** -- The full graph is reduced to highlight only the most significant edges and nodes, producing a simplified view for diagnosis.

The analysis pipeline also includes data preprocessing (normalization, FFT, band-pass filtering, detrending) and visualization of results.

## Documentation

For detailed usage guides, method descriptions, and API reference, see the [documentation on Read the Docs](https://faultmap.readthedocs.io/).

To build the documentation locally:

```bash
pip install ".[docs]"
sphinx-build -b html docs/ docs/_build/html
```

## Prerequisites

- **Python 3.11+**
- **Java JDK 8+** (required by [JIDT](https://github.com/jlizier/jidt), which computes transfer entropy)
  - The `JAVA_HOME` environment variable must point to the JDK installation directory.
- **C++ compiler** compatible with your Python version (required by some dependencies)
  - On Windows, install the `VC++ 2015.3 v14.00 (v140) toolset for desktop` from the Visual Studio installer.
- **HDF5 development libraries** (required by the `tables` dependency)
  - On Debian/Ubuntu: `sudo apt-get install libhdf5-dev pkg-config`

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/SimonStreicher/FaultMap.git
cd FaultMap
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[test]"
pytest  # Verify the installation
```

### Using conda

```bash
git clone https://github.com/SimonStreicher/FaultMap.git
cd FaultMap
conda create --name faultmap python=3.12
conda activate faultmap
pip install -e .
pytest
```

### Using Docker

A Docker image with all dependencies pre-installed is available:

```bash
docker pull simonstreicher/faultmap
```

To build the image locally, see the [FaultMapDocker repository](https://github.com/SimonStreicher/FaultMapDocker).

## Setup

After installation, create a `case_config.json` file in the project root directory that specifies paths to your data, configuration files, results directory, and the JIDT jar file.

An example `case_config.json`:

```json
{
  "data_loc": "~/faultmap/faultmap_data",
  "config_loc": "~/repos/faultmapconfigs",
  "save_loc": "~/faultmap/faultmap_results",
  "infodynamics_loc": "~/repos/FaultMap/infodynamics.jar"
}
```

| Key | Description |
|-----|-------------|
| `data_loc` | Directory containing input time series CSV files |
| `config_loc` | Directory containing case configuration JSON files |
| `save_loc` | Directory where results will be written |
| `infodynamics_loc` | Path to the `infodynamics.jar` file (a tested version is included in the repository) |

## Configuration

Each analysis case requires a set of JSON configuration files in the directory specified by `config_loc`. See the `example_configs/` directory for templates.

The following configuration files are needed:

| File | Purpose |
|------|---------|
| `config_full.json` | Specifies which cases to run and global settings |
| `config_weightcalc.json` | Transfer entropy calculation parameters |
| `config_noderank.json` | Node ranking settings |
| `config_graphreduce.json` | Graph reduction parameters |
| `config_plotting.json` | Visualization settings |

A `config_full.json` file lists the cases to process:

```json
{
  "mode": "cases",
  "write_output": true,
  "cases": [
    "tennessee_eastman"
  ]
}
```

## Execution

### Full analysis pipeline

To run all analysis steps for every case listed in `config_full.json`:

```bash
python run_full.py
```

This executes the following stages in order:

1. `run_weightcalc.py` -- Compute transfer entropy weights
2. `run_createarrays.py` -- Reconstruct result arrays
3. `run_trendextraction.py` -- Extract trends from results
4. `run_noderank.py` -- Rank nodes by centrality
5. `run_graphreduce.py` -- Reduce graphs to top edges
6. `run_plotting.py` -- Generate visualizations

Each stage can also be run independently.

### Demo scripts

The `demo/` directory contains standalone scripts demonstrating individual components:

- `demo_entropycalc.py` -- Single-signal entropy calculation
- `demo_generators.py` -- Synthetic test data generation
- `demo_leadlag_analysis.py` -- Lead/lag transfer entropy analysis
- `demo_networkranking.py` -- Network ranking with eigenvector centrality

## Development

Install development and test dependencies:

```bash
pip install -e ".[dev,test]"
```

Run the test suite:

```bash
pytest tests/ -v
```

Run the linter:

```bash
ruff check .
ruff format --check .
```

## Citation

If you use FaultMap in your research, please cite:

> Streicher, S.J. (2019). FaultMap. Zenodo. https://doi.org/10.5281/zenodo.2543739

## License

FaultMap is released under the [GPL-3.0-or-later](https://www.gnu.org/licenses/gpl-3.0.html) license.
