Supporting software
===================

FaultMap produces several types of output files. The following external tools
are recommended for viewing and analyzing these outputs.

TOPCAT -- time series plotting
------------------------------

`TOPCAT <http://www.star.bris.ac.uk/~mbt/topcat/>`_ is an interactive graphical
tool for working with tabular data. The CSV output files from FaultMap can be
imported directly.

* Use the **time series plot** format for normalized time series data outputs.
* Use the **plane plot** format for transfer entropy vs. time delay, FFT
  spectra, and similar non-time-series outputs.

Cytoscape -- graph visualization
--------------------------------

`Cytoscape <https://cytoscape.org/>`_ is an open-source platform for
visualizing complex networks. Use it to produce directed graph schematics from
the GML files that FaultMap generates.

Cytoscape supports various graph layout algorithms and visual styling options
(node size by importance, edge width by weight, etc.) that are useful for
interpreting FaultMap results.

ViTables -- HDF5 data viewing
-----------------------------

`ViTables <https://vitables.org/>`_ is a graphical tool for browsing and
editing HDF5 and PyTables files. FaultMap stores intermediate results in HDF5
format, and ViTables can be used to inspect these files.
