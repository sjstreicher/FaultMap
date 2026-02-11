Transfer entropy calculation
============================

Transfer entropy is an information-theoretic measure that quantifies the
directed flow of information between two time series. In FaultMap, transfer
entropy is used to determine causal relationships between process variables:
a high transfer entropy from signal A to signal B indicates that A provides
predictive information about B's future state beyond what B's own past provides.

FaultMap computes transfer entropy using the
`Java Information Dynamics Toolkit (JIDT) <https://github.com/jlizier/jidt>`_,
accessed via JPype. The ``infodynamics`` module wraps JIDT's transfer entropy
estimators for use from Python.

Transfer entropy is computed across a range of time delays to identify the
lag at which information transfer is strongest. A significance test using
surrogate data determines whether the measured transfer entropy is
statistically meaningful.

.. automodule:: faultmap.infodynamics
   :members:
   :no-index:
