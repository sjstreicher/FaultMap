Weight calculation
==================

Weight calculation is one of the core operations in FaultMap. It processes time
series data collected from process sensors to infer the relative importance of
connections between process elements.

The weight calculation module orchestrates the computation of transfer entropy
between all pairs of process signals and produces edge weights for the directed
graph. Key steps include:

* Loading and preprocessing time series data (normalization, optional FFT and
  band-pass filtering).
* Computing pairwise transfer entropy across a range of time delays.
* Selecting optimal delay values and computing significance thresholds.
* Writing results to HDF5 storage for downstream use by the node ranking and
  graph reduction stages.

Multiprocessing is supported to parallelize pairwise computations across
available CPU cores.

.. note::
   The following is auto-generated documentation from the ``weightcalc`` module source:

.. automodule:: faultmap.weightcalc
   :members:
   :no-index:
