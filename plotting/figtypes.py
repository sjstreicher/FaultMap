# -*- coding: utf-8 -*-
"""General graph types are defined for re-usability and consistent formatting.

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

Simple/Directional/Significance weight vs. delays for selected
variables of single scenario
    Includes the option to plot the significance threshold values obtained by
    different methods

Simple/Directional/Significance weight vs. delays for specific variable
between multiple scenarios
    Includes the option to plot the significance threshold values obtained by
    different methods

Simple/Directional/Significance weights vs. boxes for specifc variable
between multiple scenarios
    Includes the option to plot the significance threshold values obtained by
    different methods (value at optimizing delay used in actual testing
    for each box under consideration)

"""

import itertools
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ranking import data_processing
from ranking.gaincalc import WeightcalcData

import plotter
#from plotter import get_scenario_data_vectors

# Preamble
#sns.set_style('seaborn-paper')
plt.style.use(['seaborn-whitegrid', 'seaborn-paper'])

mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

markers = ['o', '*', 's', 'v', 'X', 'D', 'H']

# Label dictionaries
yaxislabel = \
    {u'cross_correlation': r'Cross correlation',
     u'absolute_transfer_entropy_kernel':
         r'Simple transfer entropy (Kernel) (bits)',
     u'directional_transfer_entropy_kernel':
         r'Directional transfer entropy (Kernel) (bits)',
     u'absolute_transfer_entropy_kraskov':
         r'Simple transfer entropy (Kraskov) (bits)',
     u'directional_transfer_entropy_kraskov':
         r'Directional transfer entropy (Kraskov) (bits)'}

linelabels = \
    {'cross_correlation': r'Correlation',
     'absolute_transfer_entropy_kernel': r'Simple TE (Kernel)',
     'directional_transfer_entropy_kernel': r'Directional TE (Kernel)',
     'absolute_transfer_entropy_kraskov': r'Simple TE (Kraskov)',
     'directional_transfer_entropy_kraskov': r'Directional TE (Kraskov)'}

fitlinelabels = \
    {'cross_correlation': r'Correlation fit',
     'absolute_transfer_entropy_kernel': r'Simple TE (Kernel) fit',
     'directional_transfer_entropy_kernel': r'Directional TE (Kernel) fit',
     'absolute_transfer_entropy_kraskov': r'Simple TE (Kraskov) fit',
     'directional_transfer_entropy_kraskov': r'Directional TE (Kraskov) fit'}


def fig_timeseries(graphdata, graph, scenario, savedir):
    """Plots time series data over time."""

    graphdata.get_settings(graph)
    graphdata.get_legendbbox(graph)
    graphdata.get_plotvars(graph)
    #graphdata.get_starttime(graph)

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
                 label=r'${}$'.format(headers[varindex]))

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
    Able to iterate through multiple box indexes and source variables.

    Provides the option to plot weight significance threshold values.

    """
    plt.close('all')

    graphdata.get_legendbbox(graph)
    graphdata.get_timeunit(graph)
    graphdata.get_boxindexes(graph)
    graphdata.get_sourcevars(graph)
    graphdata.get_destvars(graph)
    graphdata.get_sigthresholdplotting(graph)
    graphdata.get_linelabels(graph)

    # Get back from savedir to weightdata source
    # This is up to the embed type level
    weightdir = data_processing.change_dirtype(savedir, 'graphs', 'weightdata')

    # Extract current method from weightdir
    dirparts = data_processing.getfolders(weightdir)
    # The method is two folders up from the embed level
    method = dirparts[-3]

    # Select typenames based on method
    if method[:16] == 'transfer_entropy':
        typenames = []
        thresh_typenames = []
        graphdata.get_typenames(graph)
        if 'simple' in graphdata.typenames:
            typenames.append('weights_absolute')
            thresh_typenames.append('sigthresh_absolute')
        if 'directional' in graphdata.typenames:
            typenames.append('weights_directional')
            thresh_typenames.append('sigthresh_directional')
        # typenames = [
        #     'weights_absolute',
        #     'weights_directional']
        # thresh_typenames = [
        #     'sigthresh_absolute',
        #     'sigthresh_directional']
    else:
        typenames = ['weights']
        thresh_typenames = ['sigthresh']

    # Get labels
    if graphdata.linelabels:
        graphdata.get_labelformat(graph)
        labels = [graphdata.labelformat.format(linelabel)
                  for linelabel in graphdata.linelabels]
    else:
        labels = [destvar for destvar in graphdata.destvars]

    for typeindex, typename in enumerate(typenames):
        for boxindex, sourcevar in itertools.product(
                graphdata.boxindexes, graphdata.sourcevars):

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

            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)

            for destvarindex, destvar in enumerate(graphdata.destvars):
                destvarvalueindex = headers.index(destvar)
                ax.plot(valuematrix[:, 0],
                        valuematrix[:, destvarvalueindex],
                        marker=markers[destvarindex], markersize=8,
                        label=labels[destvarindex])

                label_index = list(valuematrix[:, destvarvalueindex]).index(max(valuematrix[:, destvarvalueindex]))

                ax.text(valuematrix[:, 0][label_index],
                        valuematrix[:, destvarvalueindex][label_index],
                        labels[destvarindex], ha="center", va="center", size=10,
                        bbox=bbox_props)

                if graphdata.thresholdplotting:
                    ax.plot(threshmatrix[:, 0],
                            threshmatrix[:, destvarvalueindex],
                            marker="x", markersize=4,
                            linestyle=':',
                            label=destvar + ' threshold')

            # Shrink current axis by 20%
#                box = ax.get_position()
#                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            #ax.legend(loc='center left',
            #          bbox_to_anchor=graphdata.legendbbox)

            if graphdata.axis_limits is not False:
                ax.axis(graphdata.axis_limits)
            else:
                plt.gca().set_ylim(bottom=-0.05)

            plt.savefig(
                os.path.join(savedir, '{}_{}_box{:03d}_{}.pdf'.format(
                    scenario, typename, boxindex, sourcevar)),
                bbox_inches='tight', pad_inches=0)

            # Also save as SVG to allow manual editing
            plt.savefig(
                os.path.join(savedir, '{}_{}_box{:03d}_{}.svg'.format(
                    scenario, typename, boxindex, sourcevar)),
                bbox_inches='tight', pad_inches=0, format='svg')
            plt.close()

    return None


def fig_diffscen_vs_delay(graphdata, graph, scenario, savedir):
    """Plot one variable from different scenarios.
    Assumes only a single index in varindexes.
    """

    plt.close('all')

    graphdata.get_legendbbox(graph)
    graphdata.get_timeunit(graph)
    graphdata.get_boxindexes(graph)
    graphdata.get_sourcevars(graph)
    graphdata.get_destvars(graph)
    graphdata.get_sigthresholdplotting(graph)
    graphdata.get_linelabels(graph)

    # Get x-axis values
    #    graphdata.get_xvalues(graphname)

    # Get back from savedir to weightdata source
    # This is up to the embed type level
    weightdir = data_processing.change_dirtype(savedir, 'graphs', 'weightdata')

    # Extract current method from weightdir
    dirparts = data_processing.getfolders(weightdir)
    # The method is two folders up from the embed level
    method = dirparts[-3]

    # Select typenames based on method
    if method[:16] == 'transfer_entropy':
        typenames = []
        thresh_typenames = []
        graphdata.get_typenames(graph)
        if 'simple' in graphdata.typenames:
            typenames.append('weights_absolute')
            thresh_typenames.append('sigthresh_absolute')
        if 'directional' in graphdata.typenames:
            typenames.append('weights_directional')
            thresh_typenames.append('sigthresh_directional')
        # typenames = [
        #     'weights_absolute',
        #     'weights_directional']
        # thresh_typenames = [
        #     'sigthresh_absolute',
        #     'sigthresh_directional']
    else:
        typenames = ['weights']
        thresh_typenames = ['sigthresh']

    # Get labels
    if graphdata.linelabels:
        graphdata.get_labelformat(graph)
        labels = [graphdata.labelformat.format(linelabel)
                  for linelabel in graphdata.linelabels]
    else:
        labels = [destvar for destvar in graphdata.destvars]

    for typeindex, typename in enumerate(typenames):
        for boxindex, sourcevar in itertools.product(
                graphdata.boxindexes, graphdata.sourcevars):

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

            _, headers = \
                data_processing.read_header_values_datafile(sourcefile)

            # Get valuematrices
            valuematrices = plotter.get_scenario_data_vectors(graphdata, sourcefile, scenario)

            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)

            xaxis_intervals = []
            relevant_values = []
            for scenarioindex, valuematrix in enumerate(valuematrices):
                # # Get the maximum from each valuematrix in the entry
                # # which corresponds to the common element of interest.
                #
                # # TODO: Fix this old hardcoded remnant
                # # 3 referred to the index of tau=1 for many cases involved
                # #        values = valuematrix[:, 3]
                # values = valuematrix[:, graphdata.varindexes]
                # xaxis_intervals.append(valuematrix[:, 0])
                # relevant_values.append(values)

                for destvarindex, destvar in enumerate(graphdata.destvars):
                    destvarvalueindex = headers.index(destvar)
                    ax.plot(valuematrix[:, 0],
                            valuematrix[:, destvarvalueindex],
                            marker=markers[scenarioindex], markersize=8,
                            label=labels[scenarioindex])

                    label_index = list(valuematrix[:, destvarvalueindex]).index(max(valuematrix[:, destvarvalueindex]))

                    ax.text(valuematrix[:, 0][label_index],
                            valuematrix[:, destvarvalueindex][label_index],
                            labels[scenarioindex], ha="center", va="center", size=10,
                            bbox=bbox_props)

            if graphdata.axis_limits is not False:
                ax.axis(graphdata.axis_limits)
            else:
                plt.gca().set_ylim(bottom=-0.05)

            plt.savefig(
                os.path.join(savedir, '{}_{}_box{:03d}_{}.pdf'.format(
                    scenario, typename, boxindex, sourcevar)),
                bbox_inches='tight', pad_inches=0)

            plt.close('all')

            # Also save as SVG to allow manual editing
            # plt.savefig(
            #     os.path.join(savedir, '{}_{}_box{:03d}_{}.svg'.format(
            #         scenario, typename, boxindex, sourcevar)),
            #     bbox_inches='tight', pad_inches=0, format='svg')
            # plt.close()

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


def fig_maxval_variables(graphdata, graph, scenario, savedir):
    """Generates a figure that shows dependence of method values on
    different variables in a scenario.
    Draws one line for each scenario.
    """

    # Get values for x-axis
    graphdata.get_xvalues(graph)
    graphdata.get_linelabels(graph)
    graphdata.get_legendbbox(graph)

    # Get back from savedir to trends source
    # This is up to the embed type level
    trendsdir = data_processing.change_dirtype(savedir, 'graphs', 'trends')

    # Extract current method and sigstatus from weightdir
    dirparts = data_processing.getfolders(trendsdir)
    # The method is three folders up from the embed level
    method = dirparts[-3]
    # The sigstatus is two folders up from the embed level
    sigstatus = dirparts[-2]

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
