# -*- coding: utf-8 -*-
"""
@author: Simon

This module executes all functions needed to draw desired plots.


"""
import os
import json
import data_processing

import config_setup


class GraphData(object):
    """Creates a graph object storing information required by
    graphing functions.

    """

    def __init__(self, mode, case, graphname):
        # Get file locations from configuration file
        self.graphconfig = json.load(open(os.path.join(
            dataloc, 'config_graphgen' + '.json')))

        self.case, self.method, self.scenario, \
            self.axis_limits = \
            [self.graphconfig[graphname][item] for item in
                ['case', 'method', 'scenario', 'axis_limits']]

        if not self.method[0] in ['tsdata', 'fft']:
            self.boxindex, self.sourcevar, self.sigtest = \
                [self.graphconfig[graphname][item] for item in
                    ['boxindex', 'sourcevar', 'sigtest']]
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

    def get_plotvars(self, graphname):
        self.plotvars = self.graphconfig[graphname]['plotvarnames']

    def get_starttime(self, graphname):
        self.starttime = self.graphconfig[graphname]['starttime']

    def get_frequencyunit(self, graphname):
        self.frequencyunit = self.graphconfig[graphname]['frequency_unit']

    def get_varindexes(self, graphname):
        self.varindexes = [x+1 for x in
                           self.graphconfig[graphname]['varindexes']]


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


def get_box_ranking_scores(graphdata):
    """Extract rankings scores for different variables over a range of boxes.

    Makes use of the boxrankdict as input.

    Returns a list of list, with entries in the first list referring to
    a specific node, and entries in the second list referring to a specific
    box.

    """

    importancedict_filename = importancedict_filename_template.format(
        graphdata.case, graphdata.scenario,
        graphdata.method[0])

    boxrankdict = json.load(open(importancedict_filename))
    importancelist = boxrankdict.items()

    return importancelist


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
            sourcefile = filename_sig_template.format(
                graphdata.case, graphdata.scenario,
                graphdata.method[0], box,
                sourcevar)
            valuematrix, _ = \
                data_processing.read_header_values_datafile(sourcefile)
            sourcevalues.append(valuematrix)
        valuematrices.append(sourcevalues)

    return valuematrices


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


def plotdraw(mode, case, writeoutput):
    graphdata = GraphData(mode, case, graphname)

    dataloc, saveloc = config_setup.get_locations()
    graphs_savedir = config_setup.ensure_existence(
        os.path.join(saveloc, 'graphs'), make=True)
    graph_filename_template = os.path.join(graphs_savedir, '{}.pdf')


    for plot_function, graphnames in graphs:
        for graphname in graphnames:
            # Test whether the figure already exists
            testlocation = graph_filename_template.format(graphname)
            print "Now plotting " + graphname
            if not os.path.exists(testlocation):
                plot_function(graphname)
            else:
                logging.info("The requested graph has already been drawn")


    return None

