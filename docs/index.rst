FaultMap Documentation
======================

FaultMap is a data-driven, model-free process fault detection and diagnosis tool
that uses information-theoretic measures (transfer entropy) and graph theory to
identify causal links between process elements.

It computes pairwise transfer entropy between process signals to build weighted
directed graphs, then applies network centrality algorithms to rank elements by
their influence. The result is a set of graphs and rankings that serve as
troubleshooting aids for plant-wide fault diagnosis.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   intro
   inputdataformat
   outputs
   supportingsoftware

.. toctree::
   :maxdepth: 2
   :caption: Methods

   transentropy
   weightcalc
   noderank
   graphreduce

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
