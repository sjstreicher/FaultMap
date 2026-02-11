Introduction
============

FaultMap is a data-driven, model-free process fault detection and diagnosis tool.
It identifies causal links between process elements using information-theoretic
measures (transfer entropy) and represents the resulting information flows as
directed graphs. These graphs serve as troubleshooting aids for plant-wide fault
diagnosis.

Basic approach
--------------

The basic approach follows a three-step process:

* **Weight calculation:** Transfer entropy is computed between all pairs of
  process signals to quantify directional information flow. This produces edge
  weights for a directed graph.
* **Node ranking:** Graph centrality algorithms (primarily eigenvector
  centrality) are applied to the weighted directed graph to rank nodes by their
  influence on the network.
* **Graph reduction:** The full graph is reduced to highlight only the most
  significant edges and nodes, producing a simplified view for diagnosis.

Prerequisites
-------------

* **Python 3.11+**
* **Java JDK 8+** -- required by `JIDT <https://github.com/jlizier/jidt>`_,
  which computes transfer entropy. The ``JAVA_HOME`` environment variable must
  point to the JDK installation directory.
* **C++ compiler** compatible with your Python version (required by some
  dependencies).
* **HDF5 development libraries** -- required by the ``tables`` dependency. On
  Debian/Ubuntu: ``sudo apt-get install libhdf5-dev pkg-config``.

Installation
------------

From source (using `uv <https://docs.astral.sh/uv/>`_)::

    git clone https://github.com/SimonStreicher/FaultMap.git
    cd FaultMap
    uv sync --extra test
    uv run pytest

A Docker image with all dependencies pre-installed is also available::

    docker pull simonstreicher/faultmap

Quick start
-----------

1. Create a ``case_config.json`` file in the project root specifying paths to
   your data, configuration files, results directory, and the JIDT jar file.
   See the ``tests/test_config.json`` file for reference.

2. Set up case-specific configuration files in the directory specified by
   ``config_loc``. Templates are provided in the ``example_configs/`` directory.

3. Run the full analysis pipeline::

       python run_full.py

   Or run individual stages (``run_weightcalc.py``, ``run_noderank.py``, etc.)
   as needed.
