Outputs
============

In this section the outputs made available to the user are described.

Normalised time series data
--------------

This is simply a CSV file with the same structure as the time series input file, with the data associated with each tag being normalised.
This is useful for qualitatively analysing patterns in the data using tools such as TOPCAT for easy time series plot creation.

FFT data
--------------

A CSV file with the first column containing the frequency and the columns after that the normalised magnitude of FFT results for the different tags.
Useful for identifying signals with peaks in the same region to decide on which band-gap filters to use for analysis of specific disturbances.

Band-gap filtered data
--------------

A CSV file with the same structure as the main time series input data file, except that normalised and band-gap filtered signals are presented.

Time shifted transfer entropy results
--------------

A CSV file presenting the directional (difference between forward and backwards) and absolute (forwards only) transfer entropies for each source/destination node pair.
A separate file for each source node is created.
Useful for investigating the smoothness of transfer entropy with respect to time delays, as well as the different regions that are most likely to be useful for further analysis.

A significance threshold for the zero-offset dataset is available as well.

Full ranking graph
--------------

A GML file providing a directed graph with edge weights and node importances. Self-loops are removed by default before ranking is performed.
Results for the following scenarios are provided:
*Dummies used in calculation, but suppressed in the final results.
*No dummies used in calculation.
*Dummies used and still visible.

Top edges only ranking graph
--------------

A GML file similar to that for the full ranking graphs, but only including the nodes and edges associated with the specified number of top edge weights.

Ordered node importances
--------------

CSV files providing node labels, descriptions and importance scores organised from largest to smallest for the different with results provided in the graph results discussed above.