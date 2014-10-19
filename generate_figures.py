
# -*- coding: utf-8 -*-
"""This module is used to generate figures used in the LaTeX documents
associated with this project.

The generated files can be used directly by adding to the graph folder of
the LaTeX repository.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import json

import config_setup
from ranking import data_processing

logging.basicConfig(level=logging.INFO)

dataloc, saveloc = config_setup.get_locations()
graphs_savedir = config_setup.ensure_existance(os.path.join(saveloc, 'graphs'),
                                               make=True)
graph_filename_template = os.path.join(graphs_savedir, '{}.pdf')

# Preamble

sourcedir = os.path.join(saveloc, 'weightdata')
sourcedir_normts = os.path.join(saveloc, 'normdata')
filename_template = os.path.join(sourcedir,
                                 '{}_{}_weights_{}_{}_box{:03d}_{}.csv')

sig_filename_template = os.path.join(sourcedir,
                                     '{}_{}_sigthresh_{}_box{:03d}_{}.csv')

filename_normts_template = os.path.join(sourcedir_normts,
                                        '{}_{}_normalised_data.csv')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class GraphData:
    """Creates a graph object storing information required by
    graphing functions.

    """

    def __init__(self, graphname):
        # Get file locations from configuration file
        self.graphconfig = json.load(open(os.path.join(
            dataloc, 'config_graphgen' + '.json')))

        self.case, self.method, self.scenario, \
            self.axis_limits, self.sigtest = \
            [self.graphconfig[graphname][item] for item in
                ['case', 'method', 'scenario', 'axis_limits', 'sigtest']]

        if not self.method[0] == 'tsdata':
            self.boxindex, self.sourcevar = \
                [self.graphconfig[graphname][item] for item in
                    ['boxindex', 'sourcevar']]

        if self.sigtest:
            self.sigstatus = 'sigtested'
        else:
            self.sigstatus = 'nosigtest'

    def get_xvalues(self, graphname):
        self.xvals = self.graphconfig[graphname]['xvals']

    def get_legendbbox(self, graphname):
        self.legendbbox = self.graphconfig[graphname]['legendbbox']

    def get_linelabels(self, graphname):
        self.linelabels = self.graphconfig[graphname]['linelabels']


yaxislabel = \
    {u'cross_correlation': r'Cross correlation',
     u'absolute_transfer_entropy_kernel': r'Absolute transfer entropy (Kernel) (bits)',
     u'directional_transfer_entropy_kernel': r'Directional transfer entropy (Kernel) (bits)',
     u'absolute_transfer_entropy_kraskov': r'Absolute transfer entropy (Kraskov) (nats)',
     u'directional_transfer_entropy_kraskov': r'Directional transfer entropy (Kraskov) (nats)'}


linelabels = \
    {'cross_correlation': r'Correllation',
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


def fig_timeseries(graphname):
    """Plots time series data over time."""

    graphdata = GraphData(graphname)

    sourcefile = filename_normts_template.format(graphdata.case,
                                                 graphdata.scenario)

    valuematrix, headers = \
        data_processing.read_header_values_datafile(sourcefile)

    plt.figure(1, (12, 6))
    plt.plot(valuematrix[:, 0], valuematrix[:, 1],
             marker="o", markersize=4)

    plt.ylabel('Normalised value', fontsize=14)
    plt.xlabel(r'Time (seconds)', fontsize=14)

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

    sourcefile = filename_template.format(graphdata.case, graphdata.scenario,
                                          graphdata.method[0],
                                          graphdata.sigstatus,
                                          graphdata.boxindex,
                                          graphdata.sourcevar)

    valuematrix, headers = \
        data_processing.read_header_values_datafile(sourcefile)

    plt.figure(1, (12, 6))
    taus = [0.2, 0.5, 1.0, 2.0, 5.0]
    for i, tau in enumerate(taus):
        plt.plot(valuematrix[:, 0], valuematrix[:, i + 1], marker="o",
                 markersize=4,
                 label=r'$\tau = {:1.1f}$ seconds'.format(tau))

    plt.ylabel(yaxislabel[graphdata.method[0]], fontsize=14)
    plt.xlabel(r'Delay (samples)', fontsize=14)
    plt.legend(bbox_to_anchor=graphdata.legendbbox)

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

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


def get_scenario_data_vectors(graphdata):
    """Extract value matrices from different scenarios."""

    valuematrices = []

    for scenario in graphdata.scenario:
        sourcefile = filename_template.format(
            graphdata.case, scenario,
            graphdata.method[0], graphdata.sigstatus, graphdata.boxindex,
            graphdata.sourcevar)
        valuematrix, _ = \
            data_processing.read_header_values_datafile(sourcefile)
        valuematrices.append(valuematrix)

    return valuematrices


def get_box_data_vectors(graphdata):
    """Extract value matrices from different boxes and different
    source variables.

    Returns a list of list, with entries in the first list referring to
    a specific box, and entries in the second list referring to a specific
    source variable.

    """

    valuematrices = []
    # Get number of source variables
    for box in graphdata.boxindex:
        sourcevalues = []
        for sourceindex, sourcevar in enumerate(graphdata.sourcevar):
            sourcefile = filename_template.format(
                graphdata.case, graphdata.scenario,
                graphdata.method[0], graphdata.sigstatus, box,
                sourcevar)
            valuematrix, _ = \
                data_processing.read_header_values_datafile(sourcefile)
            sourcevalues.append(valuematrix)
        valuematrices.append(sourcevalues)

    return valuematrices


def get_box_threshold_vectors(graphdata):
    """Extract significance threshold matrices from different boxes and different
    source variables.

    Returns a list of list, with entries in the first list referring to
    a specific box, and entries in the second list referring to a specific
    source variable.

    """

    valuematrices = []
    # Get number of source variables
    for box in graphdata.boxindex:
        sourcevalues = []
        for sourceindex, sourcevar in enumerate(graphdata.sourcevar):
            sourcefile = sig_filename_template.format(
                graphdata.case, graphdata.scenario,
                graphdata.method[0], box,
                sourcevar)
            valuematrix, _ = \
                data_processing.read_header_values_datafile(sourcefile)
            sourcevalues.append(valuematrix)
        valuematrices.append(sourcevalues)

    return valuematrices


def fig_diffvar_vs_delay(graphname, difvals, linetitle):

    graphdata = GraphData(graphname)
    graphdata.get_legendbbox(graphname)

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

        values = valuematrix[:, 3]
        xaxis_intervals.append(valuematrix[:, 0])
        relevant_values.append(values)

    for i, val in enumerate(difvals):
        plt.plot(xaxis_intervals[i], relevant_values[i], marker="o",
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
    plt.legend(bbox_to_anchor=[1.0, 1.0])

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None

#######################################################################
# Plot measure values vs. sample delay for range of first order time
# constants.
#######################################################################


graphs = [

#          [fig_values_vs_delays,
#           ['firstorder_noiseonly_cc_vs_delays_scen01',
#            'firstorder_noiseonly_abs_te_vs_delays_scen01',
#            'firstorder_noiseonly_dir_te_vs_delays_scen01',
#            'firstorder_sineonly_cc_vs_delays_scen01',
#            'firstorder_sineonly_abs_te_vs_delays_scen01',
#            'firstorder_sineonly_dir_te_vs_delays_scen01',
#            'firstorder_noiseandsine_cc_vs_delays_scen01',
#            'firstorder_noiseandsine_abs_te_vs_delays_scen01',
#            'firstorder_noiseandsine_dir_te_vs_delays_scen01',
# Do this for the case of no normalization as well...
# Surpressed because it provides no useful graphs
          # 'firstorder_noiseonly_cc_vs_delays_nonorm_scen01',
          # 'firstorder_noiseonly_abs_te_vs_delays_nonorm_scen01',
          # 'firstorder_noiseonly_dir_te_vs_delays_nonorm_scen01',
#            ]],


          [fig_values_vs_delays,
           ['firstorder_noiseandsine_abs_te_vs_delays_scen01',
            'firstorder_noiseandsine_dir_te_vs_delays_scen01',
            'firstorder_noiseandsine_abs_te__kraskov_vs_delays_scen01',
            'firstorder_noiseandsine_dir_te__kraskov_vs_delays_scen01',
            ]],

#######################################################################
# Plot maximum measure values vs. first order time constants
#######################################################################
#          [fig_maxval_vs_taus,
#           ['firstorder_noiseonly_cc_vs_tau_scen01',
#            'firstorder_noiseonly_te_vs_tau_scen01',
#            'firstorder_sineonly_cc_vs_tau_scen01',
#            'firstorder_sineonly_te_vs_tau_scen01',
#           ]],

#######################################################################

# Investigate effect of noise sampling rate, simulation time step and
# noise variance  on values of measures.

# Approach: Take the maximum measure value for a time constant of unity
# from three different sets that vary the parameter of interest while
# keeping the others at the value of the base case scenario.

# In order to investigate noise sampling interval, these will be
# sets 1, 2, and 3 with noise sampling intervals of
# [0.1, 0.01, 1.0] respectively.

# For the case of noise variance, these will be sets 1, 4 and 5
# with noise variances of [0.1, 0.2 and 0.5] respectively.

# For the case of simulation time step, compare sets 1 and 6

#######################################################################


#######################################################################
# Plot measure values vs. delays for range of noise
# sampling intervals.
#######################################################################
#          [lambda graphname: fig_diffvar_vs_delay(
#              graphname, [0.1, 0.01, 1.0],
#              r'Sample rate = {:1.2f} seconds'),
#           ['firstorder_noiseonly_sampling_rate_effect_abs_te',
#            'firstorder_noiseonly_sampling_rate_effect_dir_te',
#            'firstorder_noiseonly_sampling_rate_effect_cc',
#            ]],
#
#          [lambda graphname: fig_diffvar_vs_delay(graphname, [1, 10, 0.1],
#                                                  r'Frequency = {:1.2f} Hz'),
#           ['firstorder_sineonly_frequency_effect_abs_te',
#            'firstorder_sineonly_frequency_effect_dir_te',
#            'firstorder_sineonly_frequency_effect_cc',
#           ]],
# Also consider finding a lograthmic fit.

#######################################################################
# Plot measure values vs. delays for range of noise variances.
# The case where data is normalised simply confirms that the values
# are unaffected and is not of particular interest.

# This section does not produce useful graphs and
# is therefore suppressed
#######################################################################
           # [lambda graphname: fig_diffvar_vs_delay(graphname, [0.1, 0.2, 0.5],
           #                                         r'Noise variance = {:1.1f}')
           #  ['firstorder_noiseonly_noise_variance_effect_abs_te',
           #   'firstorder_noiseonly_noise_variance_effect_dir_te',
           #   'firstorder_noiseonly_noise_variance_effect_cc',

           #   'firstorder_noiseonly_noise_variance_effect_nonorm_abs_te',
           #   'firstorder_noiseonly_noise_variance_effect_nonorm_dir_te',
           #   'firstorder_noiseonly_noise_variance_effect_nonorm_cc'
           #  ]],

#######################################################################
# Plot measure values vs. delays for range of different simulation
# time steps.
#######################################################################

#           [lambda graphname: fig_diffvar_vs_delay(
#               graphname, [0.01, 0.1],
#               r'Simulation time step = {:1.2f} seconds'),
#            ['firstorder_noiseonly_sim_time_interval_effect_abs_te',
#             'firstorder_noiseonly_sim_time_interval_effect_dir_te',
#             'firstorder_noiseonly_sim_time_interval_effect_cc',
#             'firstorder_sineonly_sim_time_interval_effect_abs_te',
#             'firstorder_sineonly_sim_time_interval_effect_dir_te',
#             'firstorder_sineonly_sim_time_interval_effect_cc',
#             ]],

#######################################################################
# Plot measure values vs. delays for range of sample sizes.
#######################################################################
#           [lambda graphname: fig_diffvar_vs_delay(
#               graphname, [200, 500, 1000, 2000, 5000],
#               r'Sample size = {}'),
#            ['firstorder_noiseonly_sample_size_effect_abs_te',
#             'firstorder_noiseonly_sample_size_effect_dir_te',
#             'firstorder_noiseonly_sample_size_effect_cc',
#             'firstorder_sineonly_sample_size_effect_abs_te',
#             'firstorder_sineonly_sample_size_effect_dir_te',
#             'firstorder_sineonly_sample_size_effect_cc',
#             ]],


#######################################################################
# Plot measure values vs. delays for range of sub-sampling intervals
#######################################################################

#           [lambda graphname: fig_diffvar_vs_delay(
#               graphname, [1, 2, 5],
#               r'Sub-sampling interval = {}'),
#            ['firstorder_noiseonly_subsampling_effect_abs_te',
#             'firstorder_noiseonly_subsampling_effect_dir_te',
#             'firstorder_noiseonly_subsampling_effect_cc',
#             ]],


#graphnames = ['firstorder_noiseonly_subsampling_effect_abs_te',
#              'firstorder_noiseonly_subsampling_effect_dir_te',
#              'firstorder_noiseonly_subsampling_effect_cc']
#
#for graphname in graphnames:
#    # Test whether the figure already exists
#    testlocation = graph_filename_template.format(graphname)
#    if not os.path.exists(testlocation):
#        fig_diffvar_vs_delay(graphname, [1, 2, 5],
#                             r'Sub-sampling interval = {}')
#    else:
#        logging.info("The requested graph has already been drawn")

#######################################################################
# Plot signal over time.
#######################################################################
#            [fig_timeseries,
#             ['noiseandsine_signal_normts_scen01',
#              'noiseonly_signal_normts_scen01',
#              'sineonly_signal_normts_scen01',
#             ]],

#######################################################################
# Plot measure values vs. taus for different sub-sampling intervals
#######################################################################

#           [lambda graphname: fig_scenario_maxval_vs_taus(
#               graphname, True),
#            ['firstorder_noiseonly_cc_subsampling_vs_tau_scen01',
#             'firstorder_noiseonly_abs_te_subsampling_vs_tau_scen01',
#             'firstorder_noiseonly_dir_te_subsampling_vs_tau_scen01',
#             ]],

#           [lambda graphname: fig_scenario_maxval_vs_taus(
#               graphname, False),
#            ['firstorder_noiseonly_cc_subsampling_vs_tau_scen02',
#             'firstorder_noiseonly_abs_te_subsampling_vs_tau_scen02',
#             'firstorder_noiseonly_dir_te_subsampling_vs_tau_scen02',
#             ]],

#######################################################################
# Plot measure values vs. sampling interval / noise sample variance
# for different taus intervals
#######################################################################

#           [lambda graphname: fig_subsampling_interval_effect(
#               graphname, False),
#            ['firstorder_noiseonly_cc_subsampling_vs_interval_effect',
#             'firstorder_noiseonly_abs_te_subsampling_vs_interval_effect',
#             'firstorder_noiseonly_dir_te_subsampling_vs_interval_effect',
#             ]],


#######################################################################
# Plot measure values vs. boxindex for different sources
#######################################################################

#           [lambda graphname: fig_boxvals_differentsources(
#               graphname),
#            ['firstorder_twonoises_differenstarts_abs_te_weight_vs_box_scen01',
#             'firstorder_twonoises_differenstarts_abs_te_weight_vs_box_scen02',
#             ]],

#           [lambda graphname: fig_boxvals_differentsources_threshold(
#               graphname),
#            ['firstorder_twonoises_differenstarts_abs_te_sigtest_weight_vs_box_scen01',
#             'firstorder_twonoises_differenstarts_abs_te_sigtest_weight_vs_box_scen02',
#             ]],

#           [lambda graphname: fig_boxvals_differentsources_ewma(
#               graphname),
#            ['firstorder_twonoises_differenstarts_abs_te_weight_vs_box_scen01_ewma',
#             'firstorder_twonoises_differenstarts_abs_te_weight_vs_box_scen02_ewma',
#             ]],
#######################################################################
# Plot example of how EWMA weight benchmark affects weight changes
#######################################################################

           [lambda graphname: demo_fig_EWMA_adjusted_weights(
               graphname),
            ['demo_EWMA_adjusted_weights',
             ]],

          ]

for plot_function, graphnames in graphs:
    for graphname in graphnames:
        # Test whether the figure already exists
        testlocation = graph_filename_template.format(graphname)
        if not os.path.exists(testlocation):
            plot_function(graphname)
        else:
            logging.info("The requested graph has already been drawn")


# Template for storing difference and absolute plots from node ranking lists
#            diffplot, absplot = plot_transient_importances(variablelist,
#                                                           transientdict,
#                                                           basevaldict)
#            diffplot_filename = os.path.join(saveloc,
#                                             "{}_diffplot.pdf"
#                                             .format(scenario))
#            absplot_filename = os.path.join(saveloc,
#                                            "{}_absplot.pdf"
#                                            .format(scenario))
#            diffplot.savefig(diffplot_filename)
#            absplot.savefig(absplot_filename)
