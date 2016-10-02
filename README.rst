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

Introduction
------------

FaultMap is a data-driven model-free process fault detection and diagnosis tool.
Causal links between processes elements are identified using information theory measures (transfer entropy).
These links are then used to create a visual representation of the main flows of information (disturbances, etc.) among the process elements as a directed graph.
These directed graphs are useful as troubleshooting aids.

Network centrality algorithms are applied to determine the most
The node ranking algorithm calculates an influence score for nodes in the network based on the strength and quality of their influence on other connected nodes (eigenvector centrality).

Documentation and demonstrations still under development.

Installation
------------

.. code-block:: bash

    conda create --name faultmap python=2
    source activate faultmap
    cd ~/repos/FaultMap
    conda install -f conda_requirements.txt
    pip install -r requirements.txt
    nosetests

A Docker image is available with all necessary packages and dependencies installed.

.. code-block:: bash

    docker pull simonstreicher/faultmap

If you want to build locally, the Dockerfile can be found at https://github.com/SimonStreicher/FaultMapDocker