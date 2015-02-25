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
    lowedge_graph_filename = os.path.join(savedir, '{}_lowedge.gml')
    
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
            # Reverse graph to make follow conventional nomenclature
            original_graph = original_graph.reverse()
            # Get appropriate weight threshold for deleting edges from graph
            threshold = compute_edge_threshold(original_graph,
                                               graphreducedata.percentile)
            # Delete low value edges from graph
            lowedge_graph = delete_lowval_edges(original_graph, threshold)            
            # Get simplified graph
            simplified_graph = delete_loworder_edges(lowedge_graph)
            # Reverse simplified and lowedge graph to conform to Cytoscape styling
            # requirements
            simplified_graph = simplified_graph.reverse()
            lowedge_graph = lowedge_graph.reverse()
            # Write simplified graph to file
            if writeoutput:
                # Write simplified graph
                nx.readwrite.write_gml(simplified_graph,
                    simplified_graph_filename.format(graphreducedata.graph))
                # Write lowedge graph                
                nx.readwrite.write_gml(lowedge_graph,
                    lowedge_graph_filename.format(graphreducedata.graph))
        else:
            logging.info("The requested output is in existance")

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
            
def delete_loworder_edges(graph):
    """For each node in the graph, check to see if any childs of a child node
    is also a child of the node being investigated.
    If true, delete the edge from the parent node to the child node that
    appears as a child of a child.
    
    Also deletes all self-loops.
    
    """
    simplified_graph = graph.copy()
    weight_dict = nx.get_edge_attributes(simplified_graph, 'weight')
    
    removed_edges = []
    for node in simplified_graph.nodes_iter():
        child_list = simplified_graph.successors(node)
        for child in child_list:
            second_deg_list = simplified_graph.successors(child)
            intersection_list = [val for val in child_list if val
                                 in second_deg_list]
            for duplicate in intersection_list:
                if simplified_graph.has_edge(node, duplicate):
#                    # Only remove the edge if its weight is less than the
#                    # direct connection
#                    # Find the importance of the first order connection
#                    firstweight = weight_dict[(node, child)]
#                    # Find the importance of the second order connection
#                    # Find the number of the second degree node
#                    secondweight = weight_dict[(child, duplicate)]
#                    if firstweight < secondweight:
                    simplified_graph.remove_edge(node, duplicate)
                    removed_edge = [node, duplicate]
                    removed_edges.append(removed_edge)
    logging.info("Removed " + str(len(removed_edges)) + " edges")
    
#     Remove nodes without edges
#     Get full dictionary of in- and out-degree
    out_deg_dict = simplified_graph.out_degree()
    in_deg_dict = simplified_graph.in_degree()
    # Remove nodes with sum(out+in) degree of zero
    node_dellist = []
    for node in simplified_graph.nodes_iter():
        if (out_deg_dict[node] == 0) and (in_deg_dict[node] == 0):
            node_dellist.append(node)
    simplified_graph.remove_nodes_from(node_dellist)
    logging.info("Removed " + str(len(node_dellist)) + \
        " nodes that were left hanging")
    
    logging.info("Simplified graph has " + 
                 str(simplified_graph.number_of_nodes()) + 
                 " nodes and " + str(simplified_graph.number_of_edges()) + 
                 " edges")
    
    # Get dictionary of node names
#    names_dict = nx.get_node_attributes(simplified_graph, 'label')
#    print names_dict

    # This method should also work, but is slow
#    for source_node in [0]:
#        for destination_node in [1]:
#            pathlist = nx.all_simple_paths(simplified_graph, source_node,
#                                destination_node)
#            # Get index of longest path
#            print max(enumerate(pathlist), key = lambda tup: len(tup[1]))
    
    return simplified_graph
                    
                    
                    
    
    
    
    
    
    





