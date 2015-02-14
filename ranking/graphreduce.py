# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:42:09 2015

@author: Simon Streicher

Receives a weighted directed graph in GML format and deletes all edges
that connects nodes that are connected via some other path. Only the longest
paths are retained.

The graph should be available in the "graphs" directory in the case
data folder.
A reduced graph will have the same title as the original file with the suffix
"_simplified".

A <casename>_graphreduce.json configuration file needs to be available in the
case directory root.
"""

# Library imports
import numpy as np
import networkx as nx
import os
import json
import logging

import config_setup
dataloc, _ = config_setup.get_locations()
graphreduce_config = json.load(open(os.path.join(dataloc, 'config'
                                              '_graphreduce' + '.json')))

writeoutput = graphreduce_config['writeoutput']
mode = graphreduce_config['mode']
cases = graphreduce_config['cases']

class GraphReduceData:
    """Creates a data object from file and or function definitions for use in
    graph reduce method.

    """

    def __init__(self, mode, case):

        # Get locations from configuration file
        self.saveloc, self.casedir, _ = \
            config_setup.runsetup(mode, case)
        # Load case config file
        self.caseconfig = json.load(open(os.path.join(self.casedir, case +
                                    '_graphreduce' + '.json')))

        # Get scenarios
        self.scenarios = self.caseconfig['scenarios']
        # Get data type
        self.datatype = self.caseconfig['datatype']

    def scenariodata(self, scenario):
        """Retrieves data particular to each scenario for the case being
        investigated.

        """

        "The scenario name is: " + scenario
        self.graph = self.caseconfig[scenario]['graph']

def reducegraph(mode, case, writeoutput):
    graphreducedata = GraphReduceData(mode, case)
    
    # Get the directory to save in
    savedir = \
        config_setup.ensure_existance(
            os.path.join(graphreducedata.saveloc,
                         'graph'), make=True)
    
    graph_filename = os.path.join(savedir, '{}.gml')    
    simplified_graph_filename = os.path.join(savedir, '{}_simplified.gml')
    
    for scenario in graphreducedata.scenarios:
        logging.info("Running scenario {}".format(scenario))
        # Update scenario-specific fields graphreducedata object
        graphreducedata.scenariodata(scenario)
        
        # Test whether the 'graph_simplified' GML file  already exists
        testlocation = \
            simplified_graph_filename.format(graphreducedata.graph)
        
        if not os.path.exists(testlocation):
            # Open the original graph
            original_graph = nx.readwrite.read_gml(graph_filename)
            # Get simplified graph
            simplified_graph = deletehighorderedges(original_graph)
            # Write simplified graph to file
            nx.readwrite.write_gml(simplified_graph,
                simplified_graph_filename(graphreducedata.graph))
            

def deletehighorderedges(graph):
    
    simplified_graph = graph
    
    return simplified_graph
                    
                    
                    
    
    
    
    
    
    





