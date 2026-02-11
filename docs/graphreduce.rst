Graph reduction
===============

After node ranking produces a complete weighted directed graph, the graph
reduction module simplifies it by retaining only the most significant edges and
their associated nodes. This produces a cleaner visualization that highlights
the dominant causal pathways in the process.

The reduction is controlled by specifying the number of top-ranked edges to
retain. Nodes that are not connected by any of these top edges are removed from
the reduced graph.

Graph reduction is applied across the same scenarios used in node ranking
(with dummies suppressed, without dummies, and with dummies visible), producing
separate reduced GML files for each.

.. note::
   The following is auto-generated documentation from the ``graphreduce`` module source:

.. automodule:: faultmap.graphreduce
   :members:
   :no-index:
