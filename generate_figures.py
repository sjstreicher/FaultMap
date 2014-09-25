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

        print self.method


def yaxislabel(method):
    if method == u'cross_correlation':
        label_y = r'Cross correlation'
    if method == u'absolute_transfer_entropy':
        label_y = r'Absolute transfer entropy (bits)'
    if method == u'directional_transfer_entropy':
        label_y = r'Directional transfer entropy (bits)'

    return label_y


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

    # Test whether the figure already exists
    testlocation = graph_filename_template.format(graphname)
    if not os.path.exists(testlocation):
        plt.figure(1, (12, 6))
        plt.plot(valuematrix[:, 0], valuematrix[:, 1], marker="o",
                 markersize=4,
                 label=r'$\tau = 0.2$ seconds')
        plt.plot(valuematrix[:, 0], valuematrix[:, 2], marker="o",
                 markersize=4,
                 label=r'$\tau = 0.5$ seconds')
        plt.plot(valuematrix[:, 0], valuematrix[:, 3], marker="o",
                 markersize=4,
                 label=r'$\tau = 1.0$ seconds')
        plt.plot(valuematrix[:, 0], valuematrix[:, 4], marker="o",
                 markersize=4,
                 label=r'$\tau = 2.0$ seconds')
        plt.plot(valuematrix[:, 0], valuematrix[:, 5], marker="o",
                 markersize=4,
                 label=r'$\tau = 5.0$ seconds')

        plt.ylabel(yaxislabel(graphdata.method[0]), fontsize=14)
        plt.xlabel(r'Delay (samples)', fontsize=14)
        plt.legend(bbox_to_anchor=[0.25, 1])

        if graphdata.axis_limits is not False:
            plt.axis(graphdata.axis_limits)

        plt.savefig(graph_filename_template.format(graphname))
        plt.close()

    else:
        logging.info("The requested graph has already been drawn")

    return None


#######################################################################
# Plot measure values vs. sample delay for range of first order time
# constants.
#######################################################################

graphnames = ['firstorder_noiseonly_cc_vs_delays_scen01',
              'firstorder_noiseonly_abs_te_vs_delays_scen01',
              'firstorder_noiseonly_dir_te_vs_delays_scen01']

for graphname in graphnames:
    fig_values_vs_delays(graphname)

#######################################################################
# Plot maximum measure values vs. first order time constants
#######################################################################


def linelabels(method):
    if method == 'cross_correlation':
        label = r'Correllation'
    if method == 'absolute_transfer_entropy':
        label = r'Absolute TE'
    if method == 'directional_transfer_entropy':
        label = r'Directional TE'

    return label


def fitlinelabels(method):
    if method == 'cross_correlation':
        label = r'Correlation fit'
    if method == 'absolute_transfer_entropy':
        label = r'Absolute TE fit'
    if method == 'directional_transfer_entropy':
        label = r'Directional TE fit'

    return label


def fig_maxval_vs_taus(graphname):
    """Generates a figure that shows dependence of method values on
    time constant and delay for signal passed through
    first order transfer functions.

    """

    graphdata = GraphData(graphname)

    taus = [0.2, 0.5, 1.0, 2.0, 5.0]

    # Test whether the figure already exists
    testlocation = graph_filename_template.format(graphname)

    if not os.path.exists(testlocation):
        plt.figure(1, (12, 6))

        for method in graphdata.method:
            print method

            sourcefile = filename_template.format(
                graphdata.case, graphdata.scenario,
                method, graphdata.boxindex,
                graphdata.sourcevar)

            valuematrix, headers = \
                data_processing.read_header_values_datafile(sourcefile)

            max_values = [max(valuematrix[:, index+1]) for index in range(5)]

            fit_params = np.polyfit(np.log(taus), np.log(max_values), 1)
            fit_y = [(i*fit_params[0] + fit_params[1]) for i in np.log(taus)]

            fitted_vals = [np.exp(val) for val in fit_y]

            plt.loglog(taus, max_values, ".", marker="o", markersize=4,
                       label=linelabels(method))

            plt.loglog(taus, fitted_vals, "--", label=fitlinelabels(method))

        plt.ylabel(r'Measure value', fontsize=14)
        plt.xlabel(r'Time constant ($\tau$)', fontsize=14)
        plt.legend()

        if graphdata.axis_limits is not False:
            plt.axis(graphdata.axis_limits)

        plt.savefig(graph_filename_template.format(graphname))
        plt.close()
    else:
        logging.info("The requested graph has already been drawn")

    return None

graphnames = ['firstorder_noiseonly_cc_vs_tau_scen01',
              'firstorder_noiseonly_te_vs_tau_scen01']

for graphname in graphnames:
    fig_maxval_vs_taus(graphname)


#
## Investigate effect of noise sampling rate / simulation time step on
## values of transfer entropies
#
## Approach: Take the maximum transfer entropy for a time constant of unity
## from sets 1, 2, and 3 with noise sampling intervals of
## [0.1, 0.01, 1.0] respectively.
#
#graphname = 'firstorder_noiseonly_sampling_rate_effect'
#
## Get the different data vectors required
## Set 1 data
#method = 'absolute_transfer_entropy'
#scenario = 'noiseonly_nosubs_set1'
#sourcefile = filename()
#valuematrix_set1, _ = \
#    data_processing.read_header_values_datafile(sourcefile)
#
#scenario = 'noiseonly_nosubs_set2'
#sourcefile = filename()
#valuematrix_set2, _ = \
#    data_processing.read_header_values_datafile(sourcefile)
#
#scenario = 'noiseonly_nosubs_set3'
#sourcefile = filename()
#valuematrix_set3, _ = \
#    data_processing.read_header_values_datafile(sourcefile)
#
#method = 'directional_transfer_entropy'
#scenario = 'noiseonly_nosubs_set1'
#sourcefile = filename()
#valuematrix_set1_dir, _ = \
#    data_processing.read_header_values_datafile(sourcefile)
#
#scenario = 'noiseonly_nosubs_set2'
#sourcefile = filename()
#valuematrix_set2_dir, _ = \
#    data_processing.read_header_values_datafile(sourcefile)
#
#scenario = 'noiseonly_nosubs_set3'
#sourcefile = filename()
#valuematrix_set3_dir, _ = \
#    data_processing.read_header_values_datafile(sourcefile)
#
#
## The maximum values associated with a time constant of unity is in the
## [100][3] entry
#
#te_abs_vals = [valuematrix_set2[100][3], valuematrix_set1[100][3],
#               valuematrix_set3[100][3]]
#te_dir_vals = [valuematrix_set2_dir[100][3], valuematrix_set1_dir[100][3],
#               valuematrix_set3_dir[100][3]]
#
#sampling_intervals = [0.01, 0.1, 1.0]
#
#directional_params = np.polyfit(np.log(sampling_intervals),
#                                np.log(te_dir_vals), 1)
#
#absolute_params = np.polyfit(np.log(sampling_intervals),
#                                np.log(te_abs_vals), 1)
#
#dir_fit_y = [(i*directional_params[0] + directional_params[1])
#             for i in np.log(sampling_intervals)]
#abs_fit_y = [(i*absolute_params[0] + absolute_params[1])
#             for i in np.log(sampling_intervals)]
#
#dir_fitted_vals = [np.exp(te) for te in dir_fit_y]
#abs_fitted_vals = [np.exp(te) for te in abs_fit_y]
#
## Test whether the figure already exists
#testlocation = graph_filename(graphname)
##if not os.path.exists(testlocation):
#if not False:
#    plt.figure(1, (12, 6))
#    plt.loglog(sampling_intervals, te_abs_vals, ".", marker="o", markersize=4,
#               label=r'absolute')
#    plt.loglog(sampling_intervals, te_dir_vals, ".", marker="o", markersize=4,
#               label=r'directional')
#    plt.loglog(sampling_intervals, abs_fitted_vals, "--", markersize=4,
#               label=r'absolute fit')
#    plt.loglog(sampling_intervals, dir_fitted_vals, "--", markersize=4,
#               label=r'directional fit')
#    plt.ylabel(r'Transfer entropy (bits)', fontsize=14)
#    plt.xlabel(r'Noise sampling rate ($\frac{1}{s}$)', fontsize=14)
#    plt.legend()
#
#    plt.savefig(graph_filename(graphname))
#    plt.close()
#
#else:
#    logging.info("The requested graph has already been drawn")
#
#
## Template for storing difference and absolute plots from node ranking lists
##            diffplot, absplot = plot_transient_importances(variablelist,
##                                                           transientdict,
##                                                           basevaldict)
##            diffplot_filename = os.path.join(saveloc,
##                                             "{}_diffplot.pdf"
##                                             .format(scenario))
##            absplot_filename = os.path.join(saveloc,
##                                            "{}_absplot.pdf"
##                                            .format(scenario))
##            diffplot.savefig(diffplot_filename)
##            absplot.savefig(absplot_filename)
