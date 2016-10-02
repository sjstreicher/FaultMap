# -*- coding: utf-8 -*-
"""
General graph types are defined for re-usability and consistent formatting.

The style is selected to be compatible with greyscale print work as far
as reasonably possible.

Graph types supported include:

Time series plots
    Plots values of variables across time for specified variables
    of a single scenario

Scatter plot
    General scatter plot for plotting arbitrary data with no set formats.

FFT plot
    Plots FFT magnitude across frequency for specified variables
    of a single scenario

Absolute/Directional/Significance weight vs. delays for selected
variables of single scenario
    Includes the option to plot the significance threshold values obtained by
    different methods

Absolute/Directional/Significance weight vs. delays for specific variable
between multiple scenarios
    Includes the option to plot the significance threshold values obtained by
    different methods

Absolute/Directional/Significance weights vs. boxes for specifc variable
between multiple scenarios
    Includes the option to plot the significance threshold values obtained by
    different methods (value at optimizing delay used in actual testing
    for each box under consideration)

@author: Simon Streicher
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ranking import data_processing
from ranking.gaincalc import WeightcalcData

# Preamble
sns.set_style('darkgrid')
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
         r'Absolute transfer entropy (Kraskov) (bits)',
     u'directional_transfer_entropy_kraskov':
         r'Directional transfer entropy (Kraskov) (bits)'}

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


def fig_scatter(graphdata, graph, scenario, savedir):
#    TODO: Implement general scatter plot type
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
