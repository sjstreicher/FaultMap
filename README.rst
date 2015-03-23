FaultMap
========

FaultMap is a process fault detection/diagnosis as well as root cause analysis tool.
Causal links between processes elements are identified using information theory measures.
These links are then used to create a visual representation of the mayor flows of information (disturbances, etc.) among the process elements as a directed graph.
The ability to visually inspect and rearrange these digraphs into a hierarchical structure (using a tool such as Cytoscape) is arguably the most powerful feature currently offered.

Network centrality algorithms are applied in order to determine the most influential nodes in the network based on the strength and quantity of their influence on other nodes (and the importance of the nodes that are influenced).
 

Use the demo_all.py script to run all code actively used.

Installation
============
Since the required setup is not trivial, a Docker image is made available with all necessary packages and dependencies installed.

You can get the Docker images by

docker pull simonstreicher/faultmap

If you want to build locally, the Dockerfile can be found at https://github.com/SimonStreicher/FaultMapDocker