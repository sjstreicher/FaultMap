Node ranking
============

The node ranking module applies graph centrality algorithms to the weighted
directed graph produced by the weight calculation stage. The goal is to identify
the most influential process elements -- those that propagate the most
disturbance information through the network.

The primary algorithm used is **eigenvector centrality**, which assigns high
scores to nodes that are connected to other high-scoring nodes. This is
particularly effective for identifying root causes in process networks, because
the source of a disturbance tends to influence many other elements either
directly or indirectly.

Additional centrality measures available include betweenness centrality and
other NetworkX-supported algorithms.

Results are written as:

* GML graph files with node importance scores and edge weights.
* CSV files listing nodes ranked from most to least important.

.. note::
   The following is auto-generated documentation from the ``noderank`` module source:

.. automodule:: faultmap.noderank
   :members:
   :no-index:
