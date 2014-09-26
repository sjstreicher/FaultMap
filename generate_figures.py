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
filename_template = os.path.join(sourcedir,
                                 '{}_{}_weights_{}_box{:03d}_{}.csv')

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
            self.boxindex, self.sourcevar, self.axis_limits = \
            [self.graphconfig[graphname][item] for item in
                ['case', 'method', 'scenario', 'boxindex', 'sourcevar',
                 'axis_limits']]

    def xvalues(self, graphname):
        self.xvals = self.graphconfig[graphname]['xvals']

    def get_legendbbox(self, graphname):
        self.legendbbox = self.graphconfig[graphname]['legendbbox']

yaxislabel = \
    {u'cross_correlation': r'Cross correlation',
     u'absolute_transfer_entropy': r'Absolute transfer entropy (bits)',
     u'directional_transfer_entropy': r'Directional transfer entropy (bits)'}


linelabels = \
    {'cross_correlation': r'Correllation',
     'absolute_transfer_entropy': r'Absolute TE',
     'directional_transfer_entropy': r'Directional TE'}

fitlinelabels = \
    {'cross_correlation': r'Correlation fit',
     'absolute_transfer_entropy': r'Absolute TE fit',
     'directional_transfer_entropy': r'Directional TE fit'}


def fig_values_vs_delays(graphname):
    """Generates a figure that shows dependence of method values on
    time constant and delay for signal passed through
    first order transfer functions.

    """

    graphdata = GraphData(graphname)

    sourcefile = filename_template.format(graphdata.case, graphdata.scenario,
                                          graphdata.method[0],
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
    plt.legend(bbox_to_anchor=[0.25, 1])

    if graphdata.axis_limits is not False:
        plt.axis(graphdata.axis_limits)

    plt.savefig(graph_filename_template.format(graphname))
    plt.close()

    return None


def fig_maxval_vs_taus(graphname):
    """Generates a figure that shows dependence of method values on
    time constant and delay for signal passed through
    first order transfer functions.

    """

    graphdata = GraphData(graphname)

    # Get values of taus
    graphdata.xvalues(graphname)

    plt.figure(1, (12, 6))

    for method in graphdata.method:

        sourcefile = filename_template.format(
            graphdata.case, graphdata.scenario,
            method, graphdata.boxindex,
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


def get_data_vectors(graphdata):
    """Extract value matrices from different scenarios.

    """

    valuematrices = []

    for scenario in graphdata.scenario:
        sourcefile = filename_template.format(
            graphdata.case, scenario,
            graphdata.method[0], graphdata.boxindex,
            graphdata.sourcevar)
        valuematrix, _ = \
            data_processing.read_header_values_datafile(sourcefile)
        valuematrices.append(valuematrix)

    return valuematrices


def fig_diffvar_vs_delay(graphname, difvals, linetitle):

    graphdata = GraphData(graphname)
    graphdata.get_legendbbox(graphname)

    # Get x-axis values
#    graphdata.xvalues(graphname)

    plt.figure(1, (12, 6))

    # Get valuematrices
    valuematrices = get_data_vectors(graphdata)

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

#######################################################################
# Plot measure values vs. sample delay for range of first order time
# constants.
#######################################################################

graphnames = ['firstorder_noiseonly_cc_vs_delays_scen01',
              'firstorder_noiseonly_abs_te_vs_delays_scen01',
              'firstorder_noiseonly_dir_te_vs_delays_scen01',
              'firstorder_sineonly_cc_vs_delays_scen01',
              'firstorder_sineonly_abs_te_vs_delays_scen01',
              'firstorder_sineonly_dir_te_vs_delays_scen01']

for graphname in graphnames:
    # Test whether the figure already exists
    testlocation = graph_filename_template.format(graphname)
    if not os.path.exists(testlocation):
        fig_values_vs_delays(graphname)
    else:
        logging.info("The requested graph has already been drawn")

# Do this for the case of no normalization as well...
# Surpressed because it provides no useful graphs

#graphnames = ['firstorder_noiseonly_cc_vs_delays_nonorm_scen01',
#              'firstorder_noiseonly_abs_te_vs_delays_nonorm_scen01',
#              'firstorder_noiseonly_dir_te_vs_delays_nonorm_scen01']
#
#for graphname in graphnames:
#    # Test whether the figure already exists
#    testlocation = graph_filename_template.format(graphname)
#    if not os.path.exists(testlocation):
#        fig_values_vs_delays(graphname)
#    else:
#        logging.info("The requested graph has already been drawn")

#######################################################################
# Plot maximum measure values vs. first order time constants
#######################################################################

graphnames = ['firstorder_noiseonly_cc_vs_tau_scen01',
              'firstorder_noiseonly_te_vs_tau_scen01',
              'firstorder_sineonly_cc_vs_tau_scen01',
              'firstorder_sineonly_te_vs_tau_scen01']

for graphname in graphnames:
    # Test whether the figure already exists
    testlocation = graph_filename_template.format(graphname)
    if not os.path.exists(testlocation):
        fig_maxval_vs_taus(graphname)
    else:
        logging.info("The requested graph has already been drawn")

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

graphnames = ['firstorder_noiseonly_sampling_rate_effect_abs_te',
              'firstorder_noiseonly_sampling_rate_effect_dir_te',
              'firstorder_noiseonly_sampling_rate_effect_cc']


for graphname in graphnames:
    # Test whether the figure already exists
    testlocation = graph_filename_template.format(graphname)
    logging.info("Now drawing graph: " + graphname)
    if not os.path.exists(testlocation):
        fig_diffvar_vs_delay(graphname, [0.1, 0.01, 1.0],
                             r'Sample rate = {:1.2f} seconds')
    else:
        logging.info("The requested graph has already been drawn")

# Also consider finding a lograthmic fit.


#######################################################################
# Plot measure values vs. delays for range of noise variances.
# The case where data is normalised simply confirms that the values
# are unaffected and is not of particular interest.

# This section does not produce useful graphs and
# is therefore suppressed
#######################################################################

#graphnames = ['firstorder_noiseonly_noise_variance_effect_abs_te',
#              'firstorder_noiseonly_noise_variance_effect_dir_te',
#              'firstorder_noiseonly_noise_variance_effect_cc']
#
#for graphname in graphnames:
#    # Test whether the figure already exists
#    testlocation = graph_filename_template.format(graphname)
#    if not os.path.exists(testlocation):
#        fig_diffvar_vs_delay(graphname, [0.1, 0.2, 0.5],
#                             r'Noise variance = {:1.1f}')
#    else:
#        logging.info("The requested graph has already been drawn")
#
#graphnames = ['firstorder_noiseonly_noise_variance_effect_nonorm_abs_te',
#              'firstorder_noiseonly_noise_variance_effect_nonorm_dir_te',
#              'firstorder_noiseonly_noise_variance_effect_nonorm_cc']
#
#for graphname in graphnames:
#    # Test whether the figure already exists
#    testlocation = graph_filename_template.format(graphname)
#    if not os.path.exists(testlocation):
#        fig_diffvar_vs_delay(graphname, [0.1, 0.2, 0.5],
#                             r'Noise variance = {:1.1f}')
#    else:
#        logging.info("The requested graph has already been drawn")

#######################################################################
# Plot measure values vs. delays for range of different simulation
# time steps.
#######################################################################

graphnames = ['firstorder_noiseonly_sim_time_interval_effect_abs_te',
              'firstorder_noiseonly_sim_time_interval_effect_dir_te',
              'firstorder_noiseonly_sim_time_interval_effect_cc']

for graphname in graphnames:
    # Test whether the figure already exists
    testlocation = graph_filename_template.format(graphname)
    if not os.path.exists(testlocation):
        fig_diffvar_vs_delay(graphname, [0.01, 0.1],
                             r'Simulation time step = {:1.2f} seconds')
    else:
        logging.info("The requested graph has already been drawn")

#######################################################################
# Plot measure values vs. delays for range of sample sizes.
#######################################################################

graphnames = ['firstorder_noiseonly_sample_size_effect_abs_te',
              'firstorder_noiseonly_sample_size_effect_dir_te',
              'firstorder_noiseonly_sample_size_effect_cc']

for graphname in graphnames:
    # Test whether the figure already exists
    testlocation = graph_filename_template.format(graphname)
    if not os.path.exists(testlocation):
        fig_diffvar_vs_delay(graphname, [200, 500, 1000, 2000, 5000],
                             r'Sample size = {}')
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
