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

import os
import numpy as np
import matplotlib.pyplot as plt


from ranking import data_processing
from ranking.gaincalc import WeightcalcData

# Preamble

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

 # Label dictionaries
yaxislabel = \
    {u'cross_correlation': r'Cross correlation',
     u'absolute_transfer_entropy_kernel':
         r'Absolute transfer entropy (Kernel) (bits)',
     u'directional_transfer_entropy_kernel':
         r'Directional transfer entropy (Kernel) (bits)',
     u'absolute_transfer_entropy_kraskov':
         r'Absolute transfer entropy (Kraskov) (nats)',
     u'directional_transfer_entropy_kraskov':
         r'Directional transfer entropy (Kraskov) (nats)'}


linelabels = \
    {'cross_correlation': r'Correlation',
     'absolute_transfer_entropy_kernel': r'Absolute TE (Kernel)',
     'directional_transfer_entropy_kernel': r'Directional TE (Kernel)',
     'absolute_transfer_entropy_kraskov': r'Absolute TE (Kraskov)',
     'directional_transfer_entropy_kraskov': r'Directional TE (Kraskov)'}

fitlinelabels = \
    {'cross_correlation': r'Correlation fit',
     'absolute_transfer_entropy_kernel': r'Absolute TE (Kernel) fit',
     'directional_transfer_entropy_kernel': r'Directional TE (Kernel) fit',
     'absolute_transfer_entropy_kraskov': r'Absolute TE (Kraskov) fit',
     'directional_transfer_entropy_kraskov': r'Directional TE (Kraskov) fit'}


def fig_timeseries(graphdata, graph, scenario, savedir):
    """Plots time series data over time."""

    graphdata.get_settings(graph)
    graphdata.get_legendbbox(graph)
    graphdata.get_plotvars(graph)
    graphdata.get_starttime(graph)

    weightcalcdata = WeightcalcData(graphdata.mode, graphdata.case,
                                    False, False, False)
    weightcalcdata.setsettings(scenario, graphdata.settings)

    valuematrix = weightcalcdata.inputdata_normstep
    variables = weightcalcdata.variables

    plt.figure(1, (12, 6))

    for varname in graphdata.plotvars:
        varindex = variables.index(varname)
        plt.plot(np.asarray(weightcalcdata.timestamps),
                 valuematrix[:, varindex],
                 "-",
                 label=r'{}'.format(variables[varindex]))

    plt.ylabel('Normalised value', fontsize=14)
    plt.xlabel(r'Time (seconds)', fontsize=14)
    plt.legend(bbox_to_anchor=graphdata.legendbbox)

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

    plt.savefig(os.path.join(savedir, '{}_timeseries.pdf'.format(scenario)))
    plt.close()

    return None


def fig_fft(graphdata, graph, scenario, savedir):
    """Plots FFT over frequency range."""

    graphdata.get_legendbbox(graph)
    graphdata.get_frequencyunit(graph)
    graphdata.get_plotvars(graph)

    sourcefile = os.path.join(
        graphdata.saveloc, 'fftdata',
        '{}_{}_fft.csv'.format(graphdata.case, scenario))

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

    plt.savefig(os.path.join(savedir, '{}_fft.pdf'.format(scenario)))
    plt.close()

    return None


def fig_values_vs_delays(graphdata, graph, scenario, savedir):
    """Generates a figure that shows dependence of method values on delays.

    Constrained to a single scenario, box and source variable.

    Automatically iterates through absolute and directional weights.
    Able to iteratre through multiple box indexes and source variables.

    Provides the option to plot weight significance threshold values.

    """

    graphdata.get_legendbbox(graph)
    graphdata.get_timeunit(graph)
    graphdata.get_boxindexes(graph)
    graphdata.get_sourcevars(graph)
    graphdata.get_destvars(graph)
    graphdata.get_sigthresholdplotting(graph)

    # Get back from savedir to weightdata source
    # This is up to the embed type level
    weightdir = data_processing.change_dirtype(savedir, 'graphs', 'weightdata')

    # Extract current method from weightdir
    dirparts = data_processing.getfolders(weightdir)
    # The method is two folders up from the embed level
    method = dirparts[-3]

    # Select typenames based on method
    if method[:16] == 'transfer_entropy':
        typenames = [
            'weights_absolute',
            'weights_directional']
        thresh_typenames = [
            'sigthresh_absolute',
            'sigthresh_directional']
    else:
        typenames = ['weights']
        thresh_typenames = ['sigthresh']

    for typeindex, typename in enumerate(typenames):
        for boxindex in graphdata.boxindexes:
            for sourcevar in graphdata.sourcevars:

                fig = plt.figure(1, figsize=(12, 6))
                ax = fig.add_subplot(111)
                if len(typename) > 8:
                    yaxislabelstring = typename[8:] + '_' + method
                else:
                    yaxislabelstring = method
                ax.set_ylabel(yaxislabel[yaxislabelstring], fontsize=14)
                ax.set_xlabel(r'Delay ({})'.format(graphdata.timeunit),
                              fontsize=14)

                # Open data file and plot graph
                sourcefile = os.path.join(
                    weightdir, typename, 'box{:03d}'.format(boxindex),
                    '{}.csv'.format(sourcevar))

                valuematrix, headers = \
                    data_processing.read_header_values_datafile(sourcefile)

                if graphdata.thresholdplotting:
                    threshold_sourcefile = os.path.join(
                        weightdir, thresh_typenames[typeindex],
                        'box{:03d}'.format(boxindex),
                        '{}.csv'.format(sourcevar))

                    threshmatrix, headers = \
                        data_processing.read_header_values_datafile(
                            threshold_sourcefile)

                for destvar in graphdata.destvars:
                    destvarindex = graphdata.destvars.index(destvar)
                    ax.plot(valuematrix[:, 0],
                            valuematrix[:, destvarindex + 1],
                            marker="o", markersize=4,
                            label=destvar)

                    if graphdata.thresholdplotting:
                        ax.plot(threshmatrix[:, 0],
                                threshmatrix[:, destvarindex + 1],
                                marker="x", markersize=4,
                                linestyle=':',
                                label=destvar + ' threshold')

                # Shrink current axis by 20%
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                ax.legend(loc='center left',
                          bbox_to_anchor=graphdata.legendbbox)

                if graphdata.axis_limits is not False:
                    ax.axis(graphdata.axis_limits)

                plt.gca().set_ylim(bottom=-0.05)

                plt.savefig(os.path.join(
                    savedir,
                    '{}_{}_box{:03d}_{}.pdf'.format(
                        scenario, typename, boxindex, sourcevar)))
                plt.close()

    return None


def fig_values_vs_boxes(graphdata, graph, scenario, savedir):
    """Plots measure values for different boxes and multiple variable pairs.

    Makes use of the trend data generated by trendextraction from arrays.

    """

    graphdata.get_legendbbox(graph)
    graphdata.get_timeunit(graph)
    graphdata.get_sourcevars(graph)
    graphdata.get_destvars(graph)

    # Get back from savedir to trends source
    # This is up to the embed type level
    trendsdir = data_processing.change_dirtype(savedir, 'graphs', 'trends')

    # Extract current method and sigstatus from weightdir
    dirparts = data_processing.getfolders(trendsdir)
    # The method is three folders up from the embed level
    method = dirparts[-3]
    # The sigstatus is two folders up from the embed level
    sigstatus = dirparts[-2]

    # Select typenames based on method and sigstatus
    if method[:16] == 'transfer_entropy':

        typenames = [
            'weight_absolute_trend',
            'signtested_weight_directional_trend']
        delay_typenames = [
            'delay_absolute_trend',
            'delay_directional_trend']

        if sigstatus == 'sigtested':
            typenames.append('sigweight_absolute_trend')
            typenames.append('signtested_sigweight_directional_trend')

    else:
        typenames = ['weight_trend']
        delay_typenames = ['delay_trend']

        if sigstatus == 'sigtested':
            typenames.append('sigweight_trend')

    # Y axis label lookup dictionary

    yaxislabel_lookup = {
        'weight_absolute_trend': 'absolute',
        'signtested_weight_directional_trend': 'directional',
        'delay_absolute_trend': 'absolute',
        'delay_directional_trend': 'directional',
        'sigweight_absolute_trend': 'absolute',
        'signtested_sigweight_directional_trend': 'directional'}

    for typename in typenames:
        for sourcevar in graphdata.sourcevars:

            fig = plt.figure(1, figsize=(12, 6))
            ax = fig.add_subplot(111)
            if len(typename) > 15:
                yaxislabelstring = yaxislabel_lookup[typename] + '_' + method
            else:
                yaxislabelstring = method
            ax.set_ylabel(yaxislabel[yaxislabelstring], fontsize=14)
            ax.set_xlabel(r'Box', fontsize=14)

            # Open data file and plot graph
            sourcefile = os.path.join(trendsdir, sourcevar,
                                      '{}.csv'.format(typename))

            valuematrix, headers = \
                data_processing.read_header_values_datafile(sourcefile)

            for destvar in graphdata.destvars:
                destvarindex = graphdata.destvars.index(destvar)
                ax.plot(np.arange(len(valuematrix[:, 0])),
                        valuematrix[:, destvarindex],
                        marker="o", markersize=4,
                        label=destvar)

            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            ax.legend(loc='center left',
                      bbox_to_anchor=graphdata.legendbbox)

            if graphdata.axis_limits is not False:
                ax.axis(graphdata.axis_limits)

            plt.gca().set_ylim(bottom=-0.05)

            plt.savefig(os.path.join(
                savedir,
                '{}_{}_{}.pdf'.format(
                    scenario, typename, sourcevar)))
            plt.close()

    for delay_typename in delay_typenames:
        for sourcevar in graphdata.sourcevars:

            fig = plt.figure(1, figsize=(12, 6))
            ax = fig.add_subplot(111)
            if len(typename) > 8:
                yaxislabelstring = typename[8:] + '_' + method
            else:
                yaxislabelstring = method

            ax.set_ylabel(r'Delay ({})'.format(graphdata.timeunit),
                          fontsize=14)
            ax.set_xlabel(r'Box', fontsize=14)

            # Open data file and plot graph
            sourcefile = os.path.join(trendsdir, sourcevar,
                                      '{}.csv'.format(delay_typename))

            valuematrix, headers = \
                data_processing.read_header_values_datafile(sourcefile)

            for destvar in graphdata.destvars:
                destvarindex = graphdata.destvars.index(destvar)
                ax.plot(np.arange(len(valuematrix[:, 0])),
                        valuematrix[:, destvarindex],
                        marker="o", markersize=4,
                        label=destvar)

            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            ax.legend(loc='center left',
                      bbox_to_anchor=graphdata.legendbbox)

            if graphdata.axis_limits is not False:
                ax.axis(graphdata.axis_limits)

            plt.gca().set_ylim(bottom=-0.05)

            plt.savefig(os.path.join(
                savedir,
                '{}_{}_{}.pdf'.format(
                    scenario, delay_typename, sourcevar)))
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



