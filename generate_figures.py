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

import config_setup
from ranking import data_processing

logging.basicConfig(level=logging.INFO)

_, saveloc = config_setup.get_locations()
graphs_savedir = config_setup.ensure_existance(os.path.join(saveloc, 'graphs'),
                                               make=True)
graph_filename_template = os.path.join(graphs_savedir, '{}.pdf')

# Some settings appropriate to the next couple of figures


def filename():
    return filename_template.format(case, scenario, method, name)


def graph_filename(graphname):
    return graph_filename_template.format(graphname)

case = 'filters'
scenario = 'noiseonly_nosubs_set1'
method = 'absolute_transfer_entropy'
name = 'X 1'

sourcedir = os.path.join(saveloc, 'weightdata')
filename_template = os.path.join(sourcedir, '{}_{}_weights_{}_{}.csv')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Figure to show dependence of absolute value of transfer entropy on time
# constant of the process

sourcefile = filename()


# Load CSV file in order to create graphs
valuematrix, headers = data_processing.read_header_values_datafile(sourcefile)

# TODO: Create dictionary from tag_descriptions file and use this in the legend

graphname = 'firstorder_noiseonly_abs_scen01'


# Test whether the figure already exists
testlocation = graph_filename(graphname)
if not os.path.exists(testlocation):
    plt.figure(1, (12, 6))
    plt.plot(valuematrix[:, 0], valuematrix[:, 1], marker="o", markersize=4,
             label=r'$\tau = 0.2$ seconds')
    plt.plot(valuematrix[:, 0], valuematrix[:, 2], marker="o", markersize=4,
             label=r'$\tau = 0.5$ seconds')
    plt.plot(valuematrix[:, 0], valuematrix[:, 3], marker="o", markersize=4,
             label=r'$\tau = 1.0$ seconds')
    plt.plot(valuematrix[:, 0], valuematrix[:, 4], marker="o", markersize=4,
             label=r'$\tau = 2.0$ seconds')
    plt.plot(valuematrix[:, 0], valuematrix[:, 5], marker="o", markersize=4,
             label=r'$\tau = 5.0$ seconds')

    plt.ylabel(r'Absolute transfer entropy (bits)', fontsize=14)
    plt.xlabel(r'Delay (samples)', fontsize=14)
    plt.legend()

    plt.axis([70, 130, -0.05, 0.20])

    plt.savefig(graph_filename(graphname))
    plt.close()

else:
    logging.info("The requested graph has already been drawn")


# Now for directional transfer entropy
method = 'directional_transfer_entropy'

sourcefile = filename()
valuematrix, headers = data_processing.read_header_values_datafile(sourcefile)

graphname = 'firstorder_noiseonly_dir_scen01'

# Test whether the figure already exists
testlocation = graph_filename(graphname)
if not os.path.exists(testlocation):
    plt.figure(1, (12, 6))
    plt.plot(valuematrix[:, 0], valuematrix[:, 1], marker="o", markersize=4,
             label=r'$\tau = 0.2$ seconds')
    plt.plot(valuematrix[:, 0], valuematrix[:, 2], marker="o", markersize=4,
             label=r'$\tau = 0.5$ seconds')
    plt.plot(valuematrix[:, 0], valuematrix[:, 3], marker="o", markersize=4,
             label=r'$\tau = 1.0$ seconds')
    plt.plot(valuematrix[:, 0], valuematrix[:, 4], marker="o", markersize=4,
             label=r'$\tau = 2.0$ seconds')
    plt.plot(valuematrix[:, 0], valuematrix[:, 5], marker="o", markersize=4,
             label=r'$\tau = 5.0$ seconds')

    plt.ylabel(r'Directional transfer entropy (bits)', fontsize=14)
    plt.xlabel(r'Delay (samples)', fontsize=14)
    plt.legend()

    plt.axis([70, 130, -0.10, 0.15])

    plt.savefig(graph_filename(graphname))
    plt.close()

else:
    logging.info("The requested graph has already been drawn")

# Plot maximum transfer entropies values vs. first order time constants

method = 'directional_transfer_entropy'

sourcefile = filename()
valuematrix_dir, headers = \
    data_processing.read_header_values_datafile(sourcefile)

method = 'absolute_transfer_entropy'

sourcefile = filename()
valuematrix_abs, headers = \
    data_processing.read_header_values_datafile(sourcefile)

max_directional_te = valuematrix_dir[100][1:]
max_absolute_te = valuematrix_abs[100][1:]

taus = [0.2, 0.5, 1.0, 2.0, 5.0]

#dir_vals = [te / tau for te, tau in zip(max_directional_te, taus)]
#abs_vals = [te / tau for te, tau in zip(max_absolute_te, taus)]

dir_vals = [te for te in max_directional_te]
abs_vals = [te for te in max_absolute_te]

graphname = 'firstorder_noiseonly_te_vs_tau_logplot_scen01'

directional_params = np.polyfit(np.log(taus), np.log(max_directional_te), 1)

absolute_params = np.polyfit(np.log(taus), np.log(max_absolute_te), 1)

dir_fit_y = [(i*directional_params[0] + directional_params[1])
             for i in np.log(taus)]
abs_fit_y = [(i*absolute_params[0] + absolute_params[1])
             for i in np.log(taus)]

dir_fitted_vals = [np.exp(te) for te in dir_fit_y]
abs_fitted_vals = [np.exp(te) for te in abs_fit_y]
#dir_fitted_vals = [np.exp(te) / tau for te, tau in zip(dir_fit_y, taus)]
#abs_fitted_vals = [np.exp(te) / tau for te, tau in zip(abs_fit_y, taus)]

# Test whether the figure already exists
testlocation = graph_filename(graphname)
if not os.path.exists(testlocation):
    plt.figure(1, (12, 6))
    plt.loglog(taus, abs_vals, ".", marker="o", markersize=4,
               label=r'absolute')
    plt.loglog(taus, dir_vals, ".", marker="o", markersize=4,
               label=r'directional')
    plt.loglog(taus, abs_fitted_vals, "--",
               label=r'absolute fit')
    plt.loglog(taus, dir_fitted_vals, "--",
               label=r'directional fit')

    plt.ylabel(r'Transfer entropy (bits)', fontsize=14)
    plt.xlabel(r'Time constant ($\tau$)', fontsize=14)
    plt.legend()

    plt.savefig(graph_filename(graphname))
    plt.close()

else:
    logging.info("The requested graph has already been drawn")

# Investigate effect of noise sampling rate / simulation time step on
# values of transfer entropies

# Approach: Take the maximum transfer entropy for a time constant of unity
# from sets 1, 2, and 3 with noise sampling intervals of
# [0.1, 0.01, 1.0] respectively.

graphname = 'firstorder_noiseonly_sampling_rate_effect'

# Get the different data vectors required
# Set 1 data
method = 'absolute_transfer_entropy'
scenario = 'noiseonly_nosubs_set1'
sourcefile = filename()
valuematrix_set1, _ = \
    data_processing.read_header_values_datafile(sourcefile)

scenario = 'noiseonly_nosubs_set2'
sourcefile = filename()
valuematrix_set2, _ = \
    data_processing.read_header_values_datafile(sourcefile)

scenario = 'noiseonly_nosubs_set3'
sourcefile = filename()
valuematrix_set3, _ = \
    data_processing.read_header_values_datafile(sourcefile)

method = 'directional_transfer_entropy'
scenario = 'noiseonly_nosubs_set1'
sourcefile = filename()
valuematrix_set1_dir, _ = \
    data_processing.read_header_values_datafile(sourcefile)

scenario = 'noiseonly_nosubs_set2'
sourcefile = filename()
valuematrix_set2_dir, _ = \
    data_processing.read_header_values_datafile(sourcefile)

scenario = 'noiseonly_nosubs_set3'
sourcefile = filename()
valuematrix_set3_dir, _ = \
    data_processing.read_header_values_datafile(sourcefile)


# The maximum values associated with a time constant of unity is in the
# [100][3] entry

te_abs_vals = [valuematrix_set2[100][3], valuematrix_set1[100][3],
               valuematrix_set3[100][3]]
te_dir_vals = [valuematrix_set2_dir[100][3], valuematrix_set1_dir[100][3],
               valuematrix_set3_dir[100][3]]

sampling_intervals = [0.01, 0.1, 1.0]

directional_params = np.polyfit(np.log(sampling_intervals),
                                np.log(te_dir_vals), 1)

absolute_params = np.polyfit(np.log(sampling_intervals),
                                np.log(te_abs_vals), 1)

dir_fit_y = [(i*directional_params[0] + directional_params[1])
             for i in np.log(sampling_intervals)]
abs_fit_y = [(i*absolute_params[0] + absolute_params[1])
             for i in np.log(sampling_intervals)]

dir_fitted_vals = [np.exp(te) for te in dir_fit_y]
abs_fitted_vals = [np.exp(te) for te in abs_fit_y]

# Test whether the figure already exists
testlocation = graph_filename(graphname)
#if not os.path.exists(testlocation):
if not False:
    plt.figure(1, (12, 6))
    plt.loglog(sampling_intervals, te_abs_vals, ".", marker="o", markersize=4,
               label=r'absolute')
    plt.loglog(sampling_intervals, te_dir_vals, ".", marker="o", markersize=4,
               label=r'directional')
    plt.loglog(sampling_intervals, abs_fitted_vals, "--", markersize=4,
               label=r'absolute fit')
    plt.loglog(sampling_intervals, dir_fitted_vals, "--", markersize=4,
               label=r'directional fit')
    plt.ylabel(r'Transfer entropy (bits)', fontsize=14)
    plt.xlabel(r'Noise sampling rate ($\frac{1}{s}$)', fontsize=14)
    plt.legend()

    plt.savefig(graph_filename(graphname))
    plt.close()

else:
    logging.info("The requested graph has already been drawn")
