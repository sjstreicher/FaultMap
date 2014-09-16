# -*- coding: utf-8 -*-
"""This module is used to generate figures used in the LaTeX documents
associated with this project.

The generated files can be used directly by adding to the graph folder of
the LaTeX repository.
"""

import os
import logging
import matplotlib.pyplot as plt

import config_setup
from ranking import data_processing

logging.basicConfig(level=logging.INFO)

_, saveloc = config_setup.get_locations()
graphs_savedir = config_setup.ensure_existance(os.path.join(saveloc, 'graphs'),
                                               make=True)
graph_filename_template = os.path.join(graphs_savedir, '{}.pdf')

# Some settings appropriate to the next couple of figures


def filename(name):
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

sourcefile = filename(name)


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

    plt.ylabel(r'Absolute transfer entropy (bits)', fontsize=12)
    plt.xlabel(r'Delay (samples)', fontsize=12)
    plt.legend()

    plt.axis([70, 130, -0.05, 0.20])

    plt.savefig(graph_filename(graphname))
    plt.close()

else:
    logging.info("The requested graph has already been drawn")


# Now for directional transfer entropy
method = 'directional_transfer_entropy'

sourcefile = filename(name)
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

    plt.ylabel(r'Directional transfer entropy (bits)', fontsize=12)
    plt.xlabel(r'Delay (samples)', fontsize=12)
    plt.legend()

    plt.axis([70, 130, -0.10, 0.15])

    plt.savefig(graph_filename(graphname))
    plt.close()

else:
    logging.info("The requested graph has already been drawn")

# Plot maximum transfer entropies values vs. first order time constants

method = 'directional_transfer_entropy'

sourcefile = filename(name)
valuematrix_dir, headers = data_processing.read_header_values_datafile(sourcefile)

method = 'absolute_transfer_entropy'

sourcefile = filename(name)
valuematrix_abs, headers = data_processing.read_header_values_datafile(sourcefile)


