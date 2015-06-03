FaultMap
========

.. image:: https://travis-ci.org/SimonStreicher/FaultMap.png?branch=master
   :target: https://travis-ci.org/SimonStreicher/FaultMap
   

FaultMap is a process fault detection/diagnosis as well as root cause analysis tool.
Causal links between processes elements are identified using information theory measures.
These links are then used to create a visual representation of the mayor flows of information (disturbances, etc.) among the process elements as a directed graph.
The ability to visually inspect and rearrange these digraphs into a hierarchical structure (using a tool such as Cytoscape) is arguably the most powerful feature currently offered.

Network centrality algorithms are applied in order to determine the most influential nodes in the network based on the strength and quantity of their influence on other nodes (and the importance of the nodes that are influenced).
 
Tests and demonstrations still under development.
The correct functioning of the weight calculation component can be tested by running:

.. code:: python
  test_weightcalc.py

Installation
============
Since the required setup is not trivial, a Docker image is made available with all necessary packages and dependencies installed.
You can get the Docker image by

docker pull simonstreicher/faultmap

If you want to build locally, the Dockerfile can be found at https://github.com/SimonStreicher/FaultMapDocker