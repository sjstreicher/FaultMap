Input data formats
==================

This section describes the required and optional input data formats.

Time series data format
-----------------------

The primary input is time series data from process sensors, provided as
comma-separated value (CSV) files.

Format requirements:

* The first row must be a header line, with the first column label being
  ``Time``.
* The first column must contain timestamps in UNIX time (seconds since epoch).
* Remaining columns contain raw measurement data for each process tag -- no
  normalization is needed, as this is handled automatically during
  preprocessing.

Example::

    Time,TAG_001,TAG_002,TAG_003
    1546300800,45.2,101.3,7.81
    1546300860,45.5,101.1,7.79
    1546300920,45.3,101.4,7.82

Descriptive labels
------------------

Optional descriptive labels can be associated with each process data tag.
Provide these as a CSV file named ``tag_descriptions.csv`` with two columns:

* First column: ``Tag name``
* Second column: ``Description``

Example::

    Tag name,Description
    TAG_001,Reactor temperature
    TAG_002,Feed flow rate
    TAG_003,Product pH

These descriptions are used in output reports and graph labels to make results
more readable.

Connectivity information
------------------------

Optionally, you can constrain the analysis to only consider specific connections
between process elements. This is useful when plant topology information is
available.

.. note::
   Adding connectivity information is not always beneficial. In some cases it
   can produce poorer root cause analysis, because higher-order connections may
   play an important role in amplifying a node's centrality score. Use this
   option with care and compare results with and without connectivity
   constraints.
