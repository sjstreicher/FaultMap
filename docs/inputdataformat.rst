Input data formats
============

In this section the required and optional data input formats are described.

Time series data format
--------------

The input data format for time series data associated with data tags on the plant are comma separated files (CSV) as follows:
*The first row should be a header line, with the first column label being "Time".
*The first column should contain the time of measurements in UNIX time.
*The rest of the columns should contain the raw data - no need to be normalised, this will be done automatically in the post-processing stages.

Descriptive labels data format
--------------

Descriptive labels might be associated with each process data tag.
This should be provided in the form of a CSV file with the data tag name in the first column and the description in the second column.
The first row should have the labels "Tag name" and "Description" in them.
The filename should be``tag_descriptions.csv``.

Connectivity information
--------------

Limiting the connections to certain edges only will be an optional feature.
This is provided for the cases where plant topology information is available and considered to be important to include in the analysis.

Please note that adding connectivity information is not always helpful, and in some cases results in poorer analysis of the root cause of the problem as higher-order connections might play an important role in boosting a particular node's score in the network. 

