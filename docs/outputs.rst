Outputs
=======

This section describes the output files produced by FaultMap.

Normalized time series data
---------------------------

A CSV file with the same structure as the input time series file, where
each tag's data has been normalized. This is useful for qualitative pattern
analysis using tools such as `TOPCAT <http://www.star.bris.ac.uk/~mbt/topcat/>`_
for interactive time series plotting.

FFT data
--------

A CSV file where the first column contains frequency values and subsequent
columns contain the normalized FFT magnitude for each tag. This is useful for
identifying signals with spectral peaks in the same frequency band, which helps
determine appropriate band-pass filter settings for isolating specific
disturbances.

Band-pass filtered data
-----------------------

A CSV file with the same structure as the input time series file, containing
normalized and band-pass filtered signals. The filter parameters are specified
in the weight calculation configuration.

Transfer entropy results
------------------------

For each source node, a separate CSV file is produced containing:

* **Directional transfer entropy** -- the difference between forward and
  backward transfer entropy for each source/destination pair.
* **Absolute transfer entropy** -- the forward-only transfer entropy values.
* **Significance thresholds** -- statistical significance levels for the
  zero-offset case.

These results are presented across a range of time delays, which is useful for
investigating the smoothness of transfer entropy with respect to lag and for
identifying the most relevant time delay regions.

Full ranking graph
------------------

A GML file containing the complete directed graph with edge weights and node
importance scores. Self-loops are removed before ranking. Results are provided
for three scenarios:

* **Dummies suppressed** -- Dummy variables are included in the calculation to
  serve as significance baselines, but removed from the final graph.
* **No dummies** -- No dummy variables used in the calculation.
* **Dummies visible** -- Dummy variables are included and remain visible in the
  output graph.

These GML files can be visualized using graph tools such as
`Cytoscape <https://cytoscape.org/>`_.

Reduced ranking graph
---------------------

A GML file similar to the full ranking graph, but containing only the nodes and
edges associated with the top-ranked edge weights. This simplified view
highlights the most significant causal pathways.

Ordered node importances
------------------------

CSV files listing node labels, descriptions, and importance scores sorted from
highest to lowest. Separate files are produced for each of the graph scenarios
described above.
