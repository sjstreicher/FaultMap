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
        self.percentile = self.caseconfig[scenario]['percentile']

def reducegraph(mode, case, writeoutput):
    graphreducedata = GraphReduceData(mode, case)
    
    # Get the source directory    
    sourcedir = \
        config_setup.ensure_existance(
            os.path.join(graphreducedata.casedir,
                         'graphs'))
    
    # Get the directory to save in
    savedir = \
        config_setup.ensure_existance(
            os.path.join(graphreducedata.saveloc,
                         'graphs'), make=True)
    
    graph_filename = os.path.join(sourcedir, '{}.gml')    
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
            original_graph = nx.readwrite.read_gml(graph_filename.format(
                graphreducedata.graph))
            # Get appropriate weight threshold for deleting edges from graph
            threshold = compute_edge_threshold(original_graph,
                                               graphreducedata.percentile)
            # Delete low value edges from graph
            lowedge_graph = delete_lowval_edges(original_graph, threshold)            
            # Get simplified graph
            simplified_graph = delete_highorder_edges(lowedge_graph)
            # Write simplified graph to file
            if writeoutput:
                nx.readwrite.write_gml(simplified_graph,
                    simplified_graph_filename.format(graphreducedata.graph))

def compute_edge_threshold(graph, percentile):
    """Calculates the threshold that should be used to delete edges from the
    original graph based on determined templates.
    
    """
    # Get list of all edge weights
    weight_dict = nx.get_edge_attributes(graph, 'weight')
    edge_weightlist = []
    for edge in graph.edges_iter():
       edge_weightlist.append(weight_dict[edge])
     
    threshold = np.percentile(np.asarray(edge_weightlist), percentile)
        
    logging.info("The " + str(percentile) + "th percentile is: " + \
        str(threshold))
    return threshold
                    
def delete_lowval_edges(graph, weight_threshold):
    """Deletes all edges with weight below the threshold value.
    Also deletes all self-looping edges.
    
    """
    
    lowedge_graph = graph.copy()
    
    # First, delete all self-loops
    selfloop_list = lowedge_graph.selfloop_edges()
    lowedge_graph.remove_edges_from(selfloop_list)
    logging.info("Deleted " + str(len(selfloop_list)) + " self-looping edges")

    edge_dellist = []  
    edge_totlist = []
    weight_dict = nx.get_edge_attributes(lowedge_graph, 'weight')
    for edge in lowedge_graph.edges_iter():
        edge_totlist.append(edge)
        if weight_dict[edge] < weight_threshold:
            edge_dellist.append(edge)
    lowedge_graph.remove_edges_from(edge_dellist)
    
    logging.info("Deleted " + str(len(edge_dellist)) + "/" +
        str(graph.number_of_edges()) + " edges")
    
    return lowedge_graph
            
def delete_highorder_edges(graph):
    """For each node in the graph, check to see if any childs of a child node
    is also a child of the node being investigated.
    If true, delete the edge from the parent node to the child node that
    appears as a child of a child.
    
    Also deletes all self-loops.
    
    """
    simplified_graph = graph.copy()
    
#    for node in graph.nodes_iter():
#        print nx.info(graph, node)
    
    return simplified_graph
                    
                    
                    
    
    
    
    
    
    





