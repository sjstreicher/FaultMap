FaultMap
========

.. image:: https://travis-ci.org/SimonStreicher/FaultMap.png?branch=master
    :target: https://travis-ci.org/SimonStreicher/FaultMap
    :alt: Travis Build

.. image:: https://landscape.io/github/SimonStreicher/FaultMap/master/landscape.svg?style=flat
    :target: https://landscape.io/github/SimonStreicher/FaultMap/master
    :alt: Code Health

.. image:: https://coveralls.io/repos/github/SimonStreicher/FaultMap/badge.svg?branch=master
    :target: https://coveralls.io/github/SimonStreicher/FaultMap?branch=master
    :alt: Code Coverage

.. image:: https://codeclimate.com/github/SimonStreicher/FaultMap/badges/gpa.svg
   :target: https://codeclimate.com/github/SimonStreicher/FaultMap
   :alt: Code Climate

.. image:: https://zenodo.org/badge/14229559.svg
   :target: https://zenodo.org/badge/latestdoi/14229559
   :alt: DOI

Introduction
------------

FaultMap is a data-driven, model-free process fault detection and diagnosis tool.
Causal links between processes elements are identified using information theory measures (transfer entropy).
These links are then used to create a visual representation of the main flows of information (disturbances, etc.) among the process elements as a directed graph.
These directed graphs are useful as troubleshooting aids.

Network centrality algorithms are applied to determine the most influential elements based on the strength and quality of their influence on other connected nodes (eigenvector centrality).

Documentation and demonstrations still under development.

Installation
------------

.. code-block:: bash

    conda create --name faultmap python=2
    source activate faultmap
    cd ~/repos/FaultMap
    conda install --file conda_requirements.txt
    pip install -r requirements.txt
    nosetests

A Docker image is available with all necessary packages and dependencies installed.

.. code-block:: bash

    docker pull simonstreicher/faultmap

If you want to build locally, the Dockerfile can be found at https://github.com/SimonStreicher/FaultMapDocker

Setup
-----

Create directories for storing the data, configuration files as well as results.
Create a file ``caseconfig.json`` in the root directory, similar to ``testconfig.json`` which comes with the distribution.
Enter the full path to the data, configuration and results directories as well as the ``infodynamics.jar`` you want to use for `Java Information Dynamics Toolkit (JIDT) <https://github.com/jlizier/jidt>`_ (the tested version is included in the distribution).

Example ``caseconfig.json`` file (also included in ``example_configs`` directory):

.. code-block:: javascript

    {
      "dataloc": "~/faultmap/faultmap_data",
      "configloc": "~/repos/faultmapconfigs",
      "saveloc": "~/faultmap/faultmap_results",
      "infodynamicsloc": "~/repos/FaultMap/infodynamics.jar"
    }

Configuration
-------------

Refer to the ``example_configs`` directory in the distribution for the required format of configuration files in order to fully define cases and scenarios.
The following configuration files are needed to fully specify a specific case:

1. ``{casename}_weightcalc.json``
2. ``{casename}_noderank.json``
3. ``{casename}_graphreduce.json``
4. ``{casename}_plotting.json``

Execution
---------
In order to calculate a full set of results for a specific case, make sure this case name is included in the ``config_full.json`` file in the directory defined under ``configloc`` in the ``caseconfig.json`` file.

Example ``config_full.json`` file (also included in ``example_configs`` directory):

.. code-block:: javascript

    {
      "mode": "cases",
      "writeoutput": true,
      "cases": [
        "tennessee_eastman"
      ]
    }
