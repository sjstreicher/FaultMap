FaultMap
========

FaultMap is a process fault detection/diagnosis as well as root cause analysis tool.
Causal links between processes elements are identified using information theory measures.
These links are then used to create a visual representation of the mayor flows of information (disturbances, etc.) among the process elements as a directed graph.
The ability to visually inspect and rearrange these digraphs into a hierarchical structure ()using a tool such as Cytoscape) is arguably the most powerful feature currently offered.

Network centrality algorithms are applied in order to determine the most influential nodes in the network based on the strength and quantity of their influence on other nodes (and the importance of the nodes that are influenced).
 

Use the demo_all.py script to run all code actively used.

Installation
============
The file `requirements.txt` contains a list of requirements
which can be installed using pip as follows:

   pip install -r requirements.txt

In addition to these packages, you also need the following dependencies:

* [PyUnicorn](http://www.pik-potsdam.de/~donges/pyunicorn/installing.html) - you will have to e-mail the author for the code.
* [Java Information Dynamics Toolkit](https://code.google.com/p/information-dynamics-toolkit/) - copy the infodynamics.jar file into the repository directory.
