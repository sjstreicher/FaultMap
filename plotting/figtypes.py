# -*- coding: utf-8 -*-
"""
@author: Simon Streicher

General graph types are defined for re-usability and consistent formatting.

The style is selected to be compatible with greyscale print work as far
as reasonably possible.

Graph types supported include:


1. Time series plots
   Plots values of variables across time for specified variables
   of a single scenario

2. FFT plot
   Plots FFT magnitude across frequency for specified variables
   of a single scenario

3. Absolute/Directional/Significance weight vs. delays
   for selected variables of single scenario
   Includes the option to plot the significance threshold values obtained by
   different methods

4. Absolute/Directional/Significance weight vs. delays
   for specific variable between multiple scenarios
   Includes the option to plot the significance threshold values obtained by
   different methods

5. Absolute/Directional/Significance weights vs. boxes
   for specifc variable between multiple scenarios
   Includes the option to plot the significance threshold values obtained by
   different methods (value at optimizing delay used in actual testing
   for each box under consideration)


"""

import matplotlib.pyplot as plt
import figdatafuncs


# Preamble

plt.rc('text', usetex=True)
plt.rc('font', family='serif')



sourcedir = os.path.join(saveloc, 'weightdata')
importancedir = os.path.join(saveloc, 'noderank')
sourcedir_normts = os.path.join(saveloc, 'normdata')
sourcedir_fft = os.path.join(saveloc, 'fftdata')

filename_template = os.path.join(sourcedir,
                                 '{}_{}_weights_{}_{}_box{:03d}_{}.csv')

filename_sig_template = os.path.join(sourcedir,
                                     '{}_{}_sigthresh_{}_box{:03d}_{}.csv')

filename_normts_template = os.path.join(sourcedir_normts,
                                        '{}_{}_normalised_data.csv')

filename_fft_template = os.path.join(sourcedir_fft,
                                     '{}_{}_fft.csv')

importancedict_filename_template = os.path.join(
    importancedir,
    '{}_{}_{}_backward_rel_boxrankdict.json')




def fig_timeseries(graphname):
    """Plots time series data over time."""

    graphdata = figdatafuncs.GraphData(graphname)
    graphdata.get_legendbbox(graphname)
    graphdata.get_plotvars(graphname)
    graphdata.get_starttime(graphname)

    sourcefile = filename_normts_template.format(graphdata.case,
                                                 graphdata.scenario)

    valuematrix, headers = \
        data_processing.read_header_values_datafile(sourcefile)

    plt.figure(1, (12, 6))

    for varname in graphdata.plotvars:
        varindex = headers.index(varname)
        plt.plot(valuematrix[:, 0] - graphdata.starttime,
                 valuematrix[:, varindex],
                 "-",
                 label=r'{}'.format(headers[varindex]))

    plt.ylabel('Normalised value', fontsize=14)
    plt.xlabel(r'Time (seconds)', fontsize=14)
    plt.legend(bbox_to_anchor=graphdata.legendbbox)

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def fig_fft(graphname):
    """Plots FFT over frequency range."""

    graphdata = GraphData(graphname)
    graphdata.get_legendbbox(graphname)
    graphdata.get_frequencyunit(graphname)
    graphdata.get_plotvars(graphname)

    sourcefile = filename_fft_template.format(graphdata.case,
                                              graphdata.scenario)

    valuematrix, headers = \
        data_processing.read_header_values_datafile(sourcefile)

    plt.figure(1, (12, 6))

    for varname in graphdata.plotvars:
        varindex = headers.index(varname)
        plt.plot(valuematrix[:, 0], valuematrix[:, varindex],
                 "-",
                 label=r'{}'.format(headers[varindex]))

    plt.ylabel('Normalised value', fontsize=14)
    plt.xlabel(r'Frequency ({})'.format(graphdata.frequencyunit), fontsize=14)
    plt.legend(bbox_to_anchor=graphdata.legendbbox)

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def fig_values_vs_delays(graphname):
    """Generates a figure that shows dependence of method values on
    time constant and delay for signal passed through
    first order transfer functions.

    """

    graphdata = GraphData(graphname)
    graphdata.get_legendbbox(graphname)
    graphdata.get_linelabels(graphname)

    sourcefile = filename_template.format(graphdata.case, graphdata.scenario,
                                          graphdata.method[0],
                                          graphdata.sigstatus,
                                          graphdata.boxindex,
                                          graphdata.sourcevar)

    valuematrix, headers = \
        data_processing.read_header_values_datafile(sourcefile)

    fig = plt.figure(1, figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_ylabel(yaxislabel[graphdata.method[0]], fontsize=14)
    ax.set_xlabel(r'Delay (time units)', fontsize=14)

    taus = graphdata.linelabels
    for i, tau in enumerate(taus):
        ax.plot(valuematrix[:, 0], valuematrix[:, i + 1], marker="o",
                markersize=4,
                label=r'$\tau = {:1.1f}$ seconds'.format(tau))

    ax.legend(bbox_to_anchor=graphdata.legendbbox)
    if graphdata.axis_limits is not False:
        ax.axis(graphdata.axis_limits)

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def fig_maxval_vs_taus(graphname):
    """Generates a figure that shows dependence of method values on
    time constant for signal passed through
    first order transfer functions.

    """

    graphdata = GraphData(graphname)

    # Get values of taus
    graphdata.get_xvalues(graphname)

    plt.figure(1, (12, 6))

    for method in graphdata.method:

        sourcefile = filename_template.format(
            graphdata.case, graphdata.scenario,
            method, graphdata.sigstatus, graphdata.boxindex,
            graphdata.sourcevar)

        valuematrix, headers = \
            data_processing.read_header_values_datafile(sourcefile)

        max_values = [max(valuematrix[:, index+1]) for index in range(5)]

        fit_params = np.polyfit(np.log(graphdata.xvals), np.log(max_values), 1)
        fit_y = [(i*fit_params[0] + fit_params[1])
                 for i in np.log(graphdata.xvals)]

        fitted_vals = [np.exp(val) for val in fit_y]

        plt.loglog(graphdata.xvals, max_values, ".", marker="o", markersize=4,
                   label=linelabels[method])

        plt.loglog(graphdata.xvals, fitted_vals, "--",
                   label=fitlinelabels[method])

    plt.ylabel(r'Measure value', fontsize=14)
    plt.xlabel(r'Time constant ($\tau$)', fontsize=14)
    plt.legend()

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def fig_scenario_maxval_vs_taus(graphname, delays=False, drawfit=False):
    """Generates a figure that shows dependence of method values on
    time constant for signal passed through
    first order transfer functions.

    Draws one line for each scenario.

    """

    graphdata = GraphData(graphname)

    # Get values for x-axis
    graphdata.get_xvalues(graphname)
    graphdata.get_linelabels(graphname)
    graphdata.get_legendbbox(graphname)

    plt.figure(1, (12, 6))

    for count, scenario in enumerate(graphdata.scenario):

        sourcefile = filename_template.format(
            graphdata.case, scenario,
            graphdata.method[0], graphdata.sigstatus, graphdata.boxindex,
            graphdata.sourcevar)

        if delays:
            valuematrix, headers = \
                data_processing.read_header_values_datafile(sourcefile)
            max_values = [max(valuematrix[:, index+1])
                          for index in range(valuematrix.shape[1]-1)]
        else:
            valuematrix, headers = \
                data_processing.read_header_values_datafile(sourcefile)
            max_values = valuematrix[1:]

        plt.plot(graphdata.xvals, max_values, "--", marker="o", markersize=4,
                 label=graphdata.linelabels[count])

        if drawfit:
            graphdata.fitlinelabels(graphname)
            fit_params = np.polyfit(np.log(graphdata.xvals),
                                    np.log(max_values), 1)
            fit_y = [(i*fit_params[0] + fit_params[1])
                     for i in np.log(graphdata.xvals)]
            fitted_vals = [np.exp(val) for val in fit_y]

            plt.loglog(graphdata.xvals, fitted_vals, "--",
                       label=graphdata.fitlinelabels[count])

    plt.ylabel(yaxislabel[graphdata.method[0]], fontsize=14)
    plt.xlabel(r'Time constant ($\tau$)', fontsize=14)
    plt.legend(bbox_to_anchor=graphdata.legendbbox)

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def fig_diffvar_vs_delay(graphname, difvals, linetitle):
    """Plot many variables from a single scenario.

    Assumes only a single scenario is defined.

    """

    graphdata = GraphData(graphname)
    graphdata.get_legendbbox(graphname)
    graphdata.get_varindexes(graphname)

    # Get x-axis values
#    graphdata.get_xvalues(graphname)

    plt.figure(1, (12, 6))

    # Get valuematrices
    valuematrices = get_scenario_data_vectors(graphdata)

    xaxis_intervals = []
    relevant_values = []
    for valuematrix in valuematrices:
        # Get the maximum from each valuematrix in the entry
        # which corresponds to the common element of interest.

        # TODO: Fix this old hardcoded remnant
        # 3 referred to the index of tau=1 for many cases involved
#        values = valuematrix[:, 3]
        values = valuematrix[:, graphdata.varindexes]
        xaxis_intervals.append(valuematrix[:, 0])
        relevant_values.append(values)

#    print xaxis_intervals[0]
#    print relevant_values[0][:, 0]
    for i, val in enumerate(difvals):
        plt.plot(xaxis_intervals[0], relevant_values[0][:, i], marker="o",
                 markersize=4,
                 label=linetitle.format(val))

    plt.ylabel(yaxislabel[graphdata.method[0]], fontsize=14)
    plt.xlabel(r'Delay (seconds)', fontsize=14)
    plt.legend(bbox_to_anchor=graphdata.legendbbox)

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def fig_diffscen_vs_delay(graphname, difvals, linetitle):
    """Plot one variable from different scenarios.

    Assumes only a single index in varindexes.

    """

    graphdata = GraphData(graphname)
    graphdata.get_legendbbox(graphname)
    graphdata.get_varindexes(graphname)

    # Get x-axis values
#    graphdata.get_xvalues(graphname)

    plt.figure(1, (12, 6))

    # Get valuematrices
    valuematrices = get_scenario_data_vectors(graphdata)

    xaxis_intervals = []
    relevant_values = []
    for valuematrix in valuematrices:
        # Get the maximum from each valuematrix in the entry
        # which corresponds to the common element of interest.

        # TODO: Fix this old hardcoded remnant
        # 3 referred to the index of tau=1 for many cases involved
#        values = valuematrix[:, 3]
        values = valuematrix[:, graphdata.varindexes]
        xaxis_intervals.append(valuematrix[:, 0])
        relevant_values.append(values)

    for i, val in enumerate(difvals):
        plt.plot(xaxis_intervals[i], relevant_values[i], marker="o",
                 markersize=4,
                 label=linetitle.format(val))

    plt.ylabel(yaxislabel[graphdata.method[0]], fontsize=14)
    plt.xlabel(r'Delay (seconds)', fontsize=14)
#    plt.legend(bbox_to_anchor=graphdata.legendbbox)

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def fig_subsampling_interval_effect(graphname, delays=False):
    """Draws graphs showing the effect that different sampling intervals have
    on the causality measure values for different noise sampling rates.

    """

    # TODO: Rewrite this into general form with for loop and arguments for
    # labels, etc.

    graphdata = GraphData(graphname)
    graphdata.get_legendbbox(graphname)

    plt.figure(1, (12, 6))

    relevant_values = []
    relevant_values_2 = []

    for count, scenario in enumerate(graphdata.scenario):

        sourcefile = filename_template.format(
            graphdata.case, scenario,
            graphdata.method[0], graphdata.sigstatus, graphdata.boxindex,
            graphdata.sourcevar)

        if delays:
            valuematrix, headers = \
                data_processing.read_header_values_datafile(sourcefile)
            relevant_values = [max(valuematrix[:, index+1])
                               for index in range(9)]
        else:
            valuematrix, headers = \
                data_processing.read_header_values_datafile(sourcefile)
            relevant_values.append(valuematrix[2])
            relevant_values_2.append(valuematrix[6])

    xvals = np.linspace(0.1, 4, 40)

    plt.plot(xvals, relevant_values, "--", marker="o", markersize=4,
             label=r'$\tau = 0.05$ seconds')
    plt.plot(xvals, relevant_values_2, "--", marker="o", markersize=4,
             label=r'$\tau = 1.00$ seconds')

    plt.ylabel(yaxislabel[graphdata.method[0]], fontsize=14)
    plt.xlabel(r'Sampling interval / noise sample rate', fontsize=14)
    plt.legend(bbox_to_anchor=graphdata.legendbbox)

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def fig_boxvals_differentsources(graphname):
    """Plots the measure values for different boxes and multiple source
    variables.

    affectedindex is the index of the affected variable that is desired

    """

    graphdata = GraphData(graphname)
    graphdata.get_legendbbox(graphname)

    # Get x-axis values
    graphdata.get_xvalues(graphname)

    # Get valuematrices
    valuematrices = get_box_data_vectors(graphdata)

    plt.figure(1, (12, 6))

    for sourceindex, sourceval in enumerate(graphdata.sourcevar):

        relevant_values = []
        for valuematrix in valuematrices:
            # Extract the values corresponding to the current sourcevar
            # and the desired affectedindex
            relevant_values.append(valuematrix[sourceindex][1])

        plt.plot(graphdata.xvals, relevant_values,
                 "--", marker="o", markersize=4,
                 label=r'Source {}'.format(sourceindex+1))

    plt.ylabel(yaxislabel[graphdata.method[0]], fontsize=14)
    plt.xlabel(r'Box number', fontsize=14)
    plt.legend(bbox_to_anchor=graphdata.legendbbox)

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def fig_boxvals_differentsources_threshold(graphname):
    """Plots the measure values for different boxes and multiple source
    variables.

    affectedindex is the index of the affected variable that is desired

    """

    graphdata = GraphData(graphname)
    graphdata.get_legendbbox(graphname)

    # Get x-axis values
    graphdata.get_xvalues(graphname)

    # Get valuematrices
    valuematrices = get_box_data_vectors(graphdata)

    thresholdmatrices = get_box_threshold_vectors(graphdata)

    plt.figure(1, (12, 6))

    for sourceindex, sourceval in enumerate(graphdata.sourcevar):
        relevant_values = []
        relevant_thresholds = []
        for valuematrix, thresholdmatrix in zip(valuematrices,
                                                thresholdmatrices):
            # Extract the values corresponding to the current sourcevar
            # and the desired affectedindex
            relevant_values.append(valuematrix[sourceindex][1])
            relevant_thresholds.append(thresholdmatrix[sourceindex][1])

        plt.plot(graphdata.xvals, relevant_values,
                 "--", marker="o", markersize=4,
                 label=r'Source {}'.format(sourceindex+1))

        plt.plot(graphdata.xvals, relevant_thresholds,
                 "-", markersize=4,
                 label=r'Source {} threshold'.format(sourceindex+1))

    plt.ylabel(yaxislabel[graphdata.method[0]], fontsize=14)
    plt.xlabel(r'Box number', fontsize=14)
    plt.legend(bbox_to_anchor=graphdata.legendbbox)

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def fig_boxvals_differentsources_ewma(graphname):
    """Plots the measure values for different boxes and multiple source
    variables.

    affectedindex is the index of the affected variable that is desired

    """

    graphdata = GraphData(graphname)
    graphdata.get_legendbbox(graphname)

    # Get x-axis values
    graphdata.get_xvalues(graphname)

    # Get valuematrices
    valuematrices = get_box_data_vectors(graphdata)

    plt.figure(1, (12, 6))

    for sourceindex, sourceval in enumerate(graphdata.sourcevar):

        relevant_values = []
        for valuematrix in valuematrices:
            # Extract the values corresponding to the current sourcevar
            # and the desired affectedindex
            relevant_values.append(valuematrix[sourceindex][1])

        # Calculate EWMA benchmarked data
        ewma_benchmark = \
            data_processing.ewma_weights_benchmark(relevant_values, 0.5)

        adjusted_values = relevant_values - ewma_benchmark

        plt.plot(graphdata.xvals, adjusted_values,
                 "--", marker="o", markersize=4,
                 label=r'Source {}'.format(sourceindex+1))

    plt.ylabel(yaxislabel[graphdata.method[0]], fontsize=14)
    plt.xlabel(r'Box number', fontsize=14)
    plt.legend(bbox_to_anchor=graphdata.legendbbox)

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def demo_fig_EWMA_adjusted_weights(graphname):
    """Plots a step input weight, EWMA benchmark of step weight and difference
    between weight and benchmark for demonstration purposes.

    Hardcoded specific example
    Make general if there is any purpose for doing so.

    """

    plt.figure(1, (12, 6))

    # Create weight array with 10 zero entries followed by 30 entries of unity
    weights = np.zeros(41)
    weights[10:] = np.ones(31)

    # Get benchmark weights with alpha = 0.3
    benchmark_weights = data_processing.ewma_weights_benchmark(weights, 0.5)

    # Get difference between weights and weight benchmark
    weights_difference = weights - benchmark_weights

    plt.plot(range(len(weights)), weights,
             "--", marker="o", markersize=4,
             label="Original weights")
    plt.plot(range(len(weights)), benchmark_weights,
             "--", marker="o", markersize=4,
             label="Weight benchmark")
    plt.plot(range(len(weights)), weights_difference,
             "--", marker="o", markersize=4,
             label="Benchmark adjusted weights")

    plt.ylabel('Weight', fontsize=14)
    plt.xlabel(r'Box number', fontsize=14)
    plt.legend(bbox_to_anchor=[1.0, 0.8])
    plt.axis([0, 40, -0.1, 1.1])

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def fig_rankings_boxes(graphname):
    """Plots the ranking values for different variables over a range of boxes.

    """

    graphdata = GraphData(graphname)
    graphdata.get_legendbbox(graphname)

    # TODO: Rewrite this to get the number of boxes automatically
    # Get x-axis values
#    graphdata.get_xvalues(graphname)

    # Get list of importances

    importancelist = get_box_ranking_scores(graphdata)
    graphdata.xvals = range(len(importancelist[0][1])+1)[1:]

    plt.figure(1, (12, 6))

    for entry in importancelist:
        if max(entry[1]) >= 0.7:
            plt.plot(graphdata.xvals, entry[1],
                     "--", marker="o", markersize=4,
                     label=entry[0])

    plt.ylabel(r'Relative importance', fontsize=14)
    plt.xlabel(r'Box number', fontsize=14)
    plt.legend(bbox_to_anchor=graphdata.legendbbox)
#
#    if graphdata.axis_limits is not False:
#        plt.axis(graphdata.axis_limits)
#
    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None



