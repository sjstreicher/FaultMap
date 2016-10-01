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
import json
import logging
import os

import networkx as nx
import numpy as np

import config_setup


class GraphReduceData(object):
    """Creates a data object from file and or function definitions for use in
    graph reduce method.

    """

    def __init__(self, mode, case):

        # Get locations from configuration file
        self.saveloc, self.caseconfigloc, self.casedir, _ = \
            config_setup.runsetup(mode, case)
        # Load case config file
        self.caseconfig = json.load(open(os.path.join(
            self.caseconfigloc, case + '_graphreduce' + '.json')))

        # Get scenarios
        self.scenarios = self.caseconfig['scenarios']
        # Get data type
        self.datatype = self.caseconfig['datatype']

    def scenariodata(self, scenario):
        """Retrieves data particular to each scenario for the case being
        investigated.

        """

        self.graph = self.caseconfig[scenario]['graph']
        self.percentile = self.caseconfig[scenario]['percentile']
        self.depth = self.caseconfig[scenario]['depth']
        self.weight_discretion = self.caseconfig[scenario]['weight_discretion']

    def get_boxes(self, scenario, datadir, typename):
        boxindexes = self.caseconfig[scenario]['boxindexes']
        if boxindexes == "all":
            boxesdir = os.path.join(datadir, typename)
            boxes = next(os.walk(boxesdir))[1]
            self.boxes = range(len(boxes))
        else:
            self.boxes = boxindexes


def dographreduce(graphreducedata, scenario, datadir,
                  typename, writeoutput):

    graph_filename = '{}.gml'
    simplified_graph_filename = '{}_simplified.gml'
    lowedge_graph_filename = '{}_lowedge.gml'

    boxesdir = os.path.join(datadir, typename)
    boxes = next(os.walk(boxesdir))[1]

    for box in boxes:
        dummiesdir = os.path.join(boxesdir, box)
        dummytypes = next(os.walk(dummiesdir))[1]
        for dummytype in dummytypes:
            # Open the original graph
            original_graph = nx.readwrite.read_gml(
                os.path.join(dummiesdir, dummytype, graph_filename.format(
                    graphreducedata.graph)))
            # Get appropriate weight threshold for deleting edges from graph
            # TODO: Implement elegant way of dealing with empty graphs
            try:
                threshold = compute_edge_threshold(original_graph,
                                                   graphreducedata.percentile)
            except:
                print "Empty graph"
                break
            # Delete low value edges from graph
            lowedge_graph = delete_lowval_edges(original_graph, threshold)
            # Get simplified graph
            simplified_graph = \
                delete_loworder_edges(lowedge_graph,
                                      graphreducedata.depth,
                                      graphreducedata.weight_discretion)

            # Write simplified graph to file
            if writeoutput:
                # Write simplified graph
                nx.readwrite.write_gml(
                    simplified_graph,
                    os.path.join(
                        dummiesdir, dummytype,
                        simplified_graph_filename.format(
                             graphreducedata.graph)))
                # Write lowedge graph
                nx.readwrite.write_gml(
                    lowedge_graph,
                    os.path.join(
                        dummiesdir, dummytype,
                        lowedge_graph_filename.format(
                             graphreducedata.graph)))
    return None


def reducegraph(mode, case, writeoutput):
    graphreducedata = GraphReduceData(mode, case)

    saveloc, caseconfigdir, casedir, _ = config_setup.runsetup(mode, case)

    # Create output directory
#    savedir = \
#        config_setup.ensure_existence(
#            os.path.join(graphreducedata.saveloc,
#                         'graphs'), make=True)

    # Directory where subdirectories for scenarios will be found
    scenariosdir = os.path.join(saveloc, 'noderank', case)

    # Get list of all scenarios
    scenarios = next(os.walk(scenariosdir))[1]

    for scenario in scenarios:
        if scenario in graphreducedata.scenarios:
            logging.info("Running scenario {}".format(scenario))
            # Update scenario-specific fields graphreducedata object
            graphreducedata.scenariodata(scenario)
            print scenario
        else:
            continue

        # Iterate through every source graph that is found inside
        # the scenario's subdirectories

        methodsdir = os.path.join(scenariosdir, scenario)
        methods = next(os.walk(methodsdir))[1]
        for method in methods:
            print method
            sigtypesdir = os.path.join(methodsdir, method)
            sigtypes = next(os.walk(sigtypesdir))[1]
            for sigtype in sigtypes:
                print sigtype
                embedtypesdir = os.path.join(sigtypesdir, sigtype)
                embedtypes = next(os.walk(embedtypesdir))[1]
                for embedtype in embedtypes:
                    print embedtype
                    datadir = os.path.join(embedtypesdir, embedtype)

                    if method[:16] == 'transfer_entropy':
                            typenames = [
                                'weight_absolute',
                                'weight_directional',
                                'signtested_weight_directional']
                            if sigtype == 'sigtest':
                                typenames.append('sigweight_absolute')
                                typenames.append(
                                    'sigweight_directional')
                                typenames.append(
                                    'signtested_sigweight_directional')
                    else:
                        typenames = ['weight']
                        if sigtype == 'sigtest':
                            typenames.append('sigweight')

                    for typename in typenames:
                        print typename
                        # Start the methods here
                        dographreduce(graphreducedata, scenario, datadir,
                                      typename, writeoutput)

    return None


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

    logging.info("The " + str(percentile) + "th percentile is: " +
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
                 str(graph.number_of_edges()) + " low valued edges")

    return lowedge_graph


def remove_duplicates(intersection_list, node, upper_child,
                      simplified_graph, weight_dict, removed_edges,
                      weight_discretion):

    for duplicate in intersection_list:
        if simplified_graph.has_edge(node, duplicate):
            if weight_discretion:
                # Only remove the edge if its weight is less than the
                # weight of the connection between the child
                # and the duplicate
                # Find the importance of the first order connection
                firstweight = weight_dict[(node, duplicate)]
                # Find the importance of the second order connection
                secondweight = weight_dict[(upper_child, duplicate)]
                if firstweight < secondweight:
                    simplified_graph.remove_edge(node, duplicate)
                    removed_edge = [node, duplicate]
                    removed_edges.append(removed_edge)
            else:
                simplified_graph.remove_edge(node, duplicate)
                removed_edge = [node, duplicate]
                removed_edges.append(removed_edge)

    return simplified_graph, removed_edges


def decompose(input_, output_):
    """Decomposes (flattens) a list of lists into a simple list."""
    if type(input_) is list:
        for subitem in input_:
            decompose(subitem, output_)
    else:
        output_.append(input_)


def delete_loworder_edges(graph, max_depth, weight_discretion):
    """Returns a simplified graph with higher order connections eliminated.
    All self-loops are also deleted.

    The level up to which the search for higher order connections should be
    completed is indiciated by the 'max_depth' parameter.
    A value of 1 means that children of children will be investigated, while a
    value of 2 means that children of children of children will be included in
    the search, and so on.
    If depth is set to "full", then the search is completed until no more
    children is found.

    If the 'weight_discretion' boolean is True, a higher order connection
    between a source node and a child will not be eliminated if this connection
    weight is higher than the weight of the connection between the last
    higher-order child to the destination node under question.

    """

    simplified_graph = graph.copy()
    weight_dict = nx.get_edge_attributes(simplified_graph, 'weight')

    removed_edges = []

    for index, node in enumerate(simplified_graph.nodes_iter()):
        children_lists = []
        logging.info("Currently processing node: " + str(node) + " (" +
                     str(index + 1) + "/" +
                     str(len(simplified_graph.nodes())) + ")")
        # First create a list of lists of all childs at different degrees,
        # up to the level where no childs are returned
        morechilds = True
        depth = 0
        child_list = simplified_graph.successors(node)
        if len(child_list) != 0:
            children_lists.append(child_list)
            while morechilds:
                depth += 1
                # Flatten list of children
                children_lists_decomp = []
                decompose(children_lists[depth-1], children_lists_decomp)
                for upper_child in children_lists_decomp:
                    # Get list of childs for each child in previous layer
                    upper_child_children = \
                        simplified_graph.successors(upper_child)
                    if len(upper_child_children) != 0:
                        children_lists.append(upper_child_children)
                        intersection_list = [val for val in child_list
                                             if val in upper_child_children]
                        simplified_graph, removed_edges = \
                            remove_duplicates(intersection_list, node,
                                              upper_child,
                                              simplified_graph,
                                              weight_dict,
                                              removed_edges,
                                              weight_discretion)

                    # If no upper children were found, set morechilds to False
                    if len(upper_child_children) == 0:
                        morechilds = False
                    # If the max_depth is not "full" and is greater than the
                    # maximum depth, set morechilds to False
                    if max_depth != "full":
                        if depth > (max_depth - 1):
                            morechilds = False

    logging.info("Removed " + str(len(removed_edges)) +
                 " higher connection edges")

    # Remove nodes without edges
    # Get full dictionary of in- and out-degree
    out_deg_dict = simplified_graph.out_degree()
    in_deg_dict = simplified_graph.in_degree()
    # Remove nodes with sum(out+in) degree of zero
    node_dellist = []
    for node in simplified_graph.nodes_iter():
        if (out_deg_dict[node] == 0) and (in_deg_dict[node] == 0):
            node_dellist.append(node)
    simplified_graph.remove_nodes_from(node_dellist)
    logging.info("Removed " + str(len(node_dellist)) +
                 " nodes that were left hanging")

    logging.info("Simplified graph has " +
                 str(simplified_graph.number_of_nodes()) +
                 " nodes and " + str(simplified_graph.number_of_edges()) +
                 " edges")

    return simplified_graph
