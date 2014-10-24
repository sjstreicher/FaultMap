# -*- coding: utf-8 -*-
"""This module is used to rank nodes in a digraph.
It requires a connection as well as a gain matrix as inputs.

Future versions will make use of an intrinsic node importance score vector (for
example, individual loop key performance indicators) as well.

@author Simon Streicher, St. Elmo Wilken

"""
# Standard libraries
import os
import json
import logging
import csv
import networkx as nx
import numpy as np
import operator
import itertools
import fnmatch

# Own libraries
import data_processing
import config_setup
import ranking

# Networks for tests
import networkgen


class NoderankData:
    """Creates a data object from file and or function definitions for use in
    weight calculation methods.

    """

    def __init__(self, mode, case):

        # Get locations from configuration file
        self.saveloc, self.casedir, _ = \
            config_setup.runsetup(mode, case)
        # Load case config file
        self.caseconfig = json.load(open(os.path.join(self.casedir, case +
                                    '_noderank' + '.json')))

        # Get scenarios
        self.scenarios = self.caseconfig['scenarios']
        # Get weight_methods
        self.weight_methods = self.caseconfig['weight_methods']
        # Get ranking methods
        self.rank_methods = self.caseconfig['rank_methods']
        # Get data type
        self.datatype = self.caseconfig['datatype']

    def scenariodata(self, scenario):
        """Retrieves data particular to each scenario for the case being
        investigated.

        """
        print "The scenario name is: " + scenario
        settings_name = self.caseconfig[scenario]['settings']
        self.connections_used = (self.caseconfig[settings_name]
                                 ['use_connections'])

        self.m = self.caseconfig[scenario]['m']
        self.alpha = self.caseconfig[scenario]['alpha']
        self.dummies = self.caseconfig[settings_name]['dummies']

        if self.datatype == 'file':
            # Retrieve connection matrix criteria from settings
            if self.connections_used:
                # Get connection (adjacency) matrix
                connectionloc = os.path.join(self.casedir, 'connections',
                                             self.caseconfig[scenario]
                                             ['connections'])
                self.connectionmatrix, self.variablelist = \
                    data_processing.read_connectionmatrix(connectionloc)

            # Get the gain matrices directory
            self.gainloc = os.path.join(self.casedir, 'gainmatrix')

        elif self.datatype == 'function':
            # Get variables, connection matrix and gainmatrix
            network_gen = self.caseconfig[scenario]['networkgen']
            self.connectionmatrix, self.gainmatrix, \
                self.variablelist, _ = \
                eval('networkgen.' + network_gen)()

        logging.info("Number of tags: {}".format(len(self.variablelist)))


def writecsv_looprank(filename, items):
    with open(filename, 'wb') as f:
        csv.writer(f).writerows(items)


def norm_dict(dictionary):
    total = sum(dictionary.values())
    # NOTE: if this is slow in Python 2, replace .items with .iteritems
    return {key: value/total for key, value in dictionary.items()}


def calc_simple_rank(gainmatrix, variables, noderankdata, rank_method,
                     package='networkx'):
    """Constructs the ranking dictionary using the eigenvector approach
    i.e. Ax = x where A is the local gain matrix.

    Taking the absolute of the gainmatrix and normalizing to conform to
    original LoopRank idea.

    """
    # Transpose gainmatrix so that we are looking at the backwards
    # ranking problem
    gainmatrix = gainmatrix.T
    # Length of gain matrix = number of nodes
    n = gainmatrix.shape[0]
    gainmatrix = np.asmatrix(gainmatrix, dtype=float)

    # Normalize the gainmatrix columns
    for col in range(n):
        colsum = np.sum(abs(gainmatrix[:, col]))
        if colsum == 0:
            # Option :1 do nothing
            None
            # Option 2: equally connect to all other nodes
    #        for row in range(n):
    #            gainmatrix[row, col] = (1. / n)
        else:
            gainmatrix[:, col] = (gainmatrix[:, col] / colsum)

    # The reset matrix is also refferred to as the personalisation matrix
    resetmatrix = np.ones((n, n), dtype=float)/n
    m = noderankdata.m
    alpha = noderankdata.alpha
    reset_gainmatrix = (m * gainmatrix) + ((1-m) * resetmatrix)

    # Normalize the reset_weightmatrix columns
    for col in range(n):
        reset_gainmatrix[:, col] = (reset_gainmatrix[:, col]
                                    / np.sum(abs(reset_gainmatrix[:, col])))

    if package == 'networkx':
        # Calculate rankings using networkx methods
        reset_gaingraph = nx.DiGraph()
        sparse_gaingraph = nx.DiGraph()

        for col, colvar in enumerate(variables):
            for row, rowvar in enumerate(variables):
                # Create fully connected weighted graph for use with
                # eigenvector centrality analysis
                reset_gaingraph.add_edge(rowvar, colvar,
                                         weight=reset_gainmatrix[row, col])
                # Create sparsely connected graph based on
                # significant edge weights
                # only for use with Katz centrality analysis
                if (gainmatrix[row, col] != 0.):
                    # The node order is source, sink according to the
                    # convention that columns are sources and rows are sinks
                    sparse_gaingraph.add_edge(rowvar, colvar,
                                              weight=gainmatrix[row, col])

        if rank_method == 'eigenvector':
            eig_rankingdict = nx.eigenvector_centrality(
                reset_gaingraph.reverse())
            eig_rankingdict_norm = norm_dict(eig_rankingdict)
            rankingdict = eig_rankingdict_norm

        elif rank_method == 'katz':
            katz_rankingdict = nx.katz_centrality(sparse_gaingraph.reverse(),
                                                  alpha)
            katz_rankingdict_norm = norm_dict(katz_rankingdict)
            rankingdict = katz_rankingdict_norm

        elif rank_method == 'pagerank':
            pagerank_rankingdict = nx.pagerank(sparse_gaingraph.reverse(),
                                               m)
            pagerank_rankingdict_norm = norm_dict(pagerank_rankingdict)
            rankingdict = pagerank_rankingdict_norm
#        else:
            # TODO: Raise exception stating that method is not defined

    elif package == 'simple':
        if rank_method == 'eigenvector':
            # Calculate rankings using own straightforward implementation
            # Only for development and confirmation purposes that networkx is
            # being used correctly.

            [eigval, eigvec] = np.linalg.eig(reset_gainmatrix)
#           [eigval_gain, eigvec_gain] = np.linalg.eig(gainmatrix)
            maxeigindex = np.argmax(eigval)
            rankarray = eigvec[:, maxeigindex]
            rankarray_list = [rankelement[0, 0] for rankelement in rankarray]
            # Take absolute values of ranking values
            rankarray = abs(np.asarray(rankarray_list))
            # This is the 1-dimensional array composed of rankings (normalised)
            rankarray_norm = (1 / sum(rankarray)) * rankarray
            # Create a dictionary of the rankings with their respective nodes
            # i.e. {NODE:RANKING}
            rankingdict = dict(zip(variables, rankarray_norm))
#        else:
            # TODO: Raise exception stating that method is not defined

    rankinglist = sorted(rankingdict.iteritems(),
                         key=operator.itemgetter(1),
                         reverse=True)

#    nx.write_gml(reset_gaingraph, os.path.join(noderankdata.saveloc,
#                 "reset_gaingraph.gml"))
#    nx.write_gml(sparse_gaingraph, os.path.join(noderankdata.saveloc,
#                 "sparse_gaingraph.gml"))

    return rankingdict, rankinglist


def normalise_rankinglist(rankingdict, originalvariables):
    normalised_rankingdict = dict()
    for variable in originalvariables:
        normalised_rankingdict[variable] = rankingdict[variable]

    # Normalise rankings
    total = sum(normalised_rankingdict.values())
    for variable in originalvariables:
        normalised_rankingdict[variable] = \
            normalised_rankingdict[variable] / total

    normalised_rankinglist = sorted(normalised_rankingdict.iteritems(),
                                    key=operator.itemgetter(1),
                                    reverse=True)

    return normalised_rankinglist


def calc_transient_importancediffs(rankingdicts, variablelist):
    """Returns three dictionaries with importance scores that can be used in
    further analysis.

    transientdict is a dictionary with a vector of successive differences in
    importance scores between boxes for each variable entry - it's first entry
    is the difference in importance between box002 - box001 and will be empty
    if there is only a single dictionary in the rankingdicts input

    basevaldict contains the absolute value of the first box - when added
    to the transientdict it can be used to reconstruct the importance
    scores for each box

    boxrankdict simply lists the importance scores for each box in a vector
    associated with each variable

    """
    transientdict = dict()
    basevaldict = dict()
    boxrankdict = dict()
    for variable in variablelist:
        diffvect = np.empty((1, len(rankingdicts)-1))[0]
        rankvect = np.empty((1, len(rankingdicts)))[0]
        rankvect[:] = np.NAN
        diffvect[:] = np.NAN
        basevaldict[variable] = rankingdicts[0][variable]
        # Get initial previous importance
        prev_importance = basevaldict[variable]
        for index, rankingdict in enumerate(rankingdicts[1:]):
            diffvect[index] = rankingdict[variable] - prev_importance
            prev_importance = rankingdict[variable]
        transientdict[variable] = diffvect.tolist()
        for index, rankingdict in enumerate(rankingdicts[:]):
            rankvect[index] = rankingdict[variable]
        boxrankdict[variable] = rankvect.tolist()

    return transientdict, basevaldict, boxrankdict


def create_importance_graph(variablelist, closedconnections,
                            openconnections, gainmatrix, ranks):
    """Generates a graph containing the
    connectivity and importance of the system being displayed.
    Edge Attribute: color for control connection
    Node Attribute: node importance

    """

    opengraph = nx.DiGraph()

    # Verify why these indexes are switched and correct
    for col, row in itertools.izip(openconnections.nonzero()[1],
                                   openconnections.nonzero()[0]):

        opengraph.add_edge(variablelist[col], variablelist[row],
                           weight=gainmatrix[row, col])
    openedgelist = opengraph.edges()

    closedgraph = nx.DiGraph()
    for col, row in itertools.izip(closedconnections.nonzero()[1],
                                   closedconnections.nonzero()[0]):
        newedge = (variablelist[col], variablelist[row])
        closedgraph.add_edge(*newedge, weight=gainmatrix[row, col],
                             controlloop=int(newedge not in openedgelist))

    for node in closedgraph.nodes():
        closedgraph.add_node(node, importance=ranks[node])

    return closedgraph, opengraph


def gainmatrix_preprocessing(gainmatrix):
    """Moves the mean and scales the variance of the elements in the
    gainmatrix to a specified value.

    Only operates on nonzero weights.

    INCOMPLETE

    """

    # Modify the gainmatrix to have a specific mean
    # Should only be used for development analysis - generally
    # destroys information.
    # Not sure what effect will be if data is variance scaled as well.

    # Get the mean of the samples in the gainmatrix that correspond
    # to the desired connectionmatrix.
    counter = 0
    gainsum = 0
    for col, row in itertools.izip(gainmatrix.nonzero()[0],
                                   gainmatrix.nonzero()[1]):
        gainsum += gainmatrix[col, row]
        counter += 1

    currentmean = gainsum / counter
    meanscale = 1. / currentmean

    # Write meandiff to all gainmatrix elements indicated by connectionmatrix
    modgainmatrix = np.zeros_like(gainmatrix)

    for col, row in itertools.izip(gainmatrix.nonzero()[0],
                                   gainmatrix.nonzero()[1]):
        modgainmatrix[col, row] = gainmatrix[col, row] * meanscale

    return modgainmatrix, currentmean


def calc_gainrank(gainmatrix, noderankdata, rank_method,
                  dummyweight):
    """Calculates backward rankings.

    """

    backwardconnection, backwardgain, backwardvariablelist = \
        data_processing.rankbackward(noderankdata.variablelist, gainmatrix,
                                     noderankdata.connectionmatrix,
                                     dummyweight, noderankdata.dummies)

    backwardrankingdict, backwardrankinglist = \
        calc_simple_rank(backwardgain, backwardvariablelist, noderankdata,
                         rank_method)

    rankingdicts = [backwardrankingdict]
    rankinglists = [backwardrankinglist]
    connections = [backwardconnection]
    variables = [backwardvariablelist]
    gains = [np.array(backwardgain)]

    return rankingdicts, rankinglists, connections, variables, gains


def get_gainmatrices(noderankdata, countlocation, gainmatrix_filename,
                     case, scenario, method):
    """Searches in countlocation for all gainmatrices CSV files
    associated with the specific case, scenario and method at hand and
    then returns all relevant gainmatrices in a list which can be used to
    calculate the change of importances over time (transient importances).

    """
    # Store all relevant gainmatrices in a list
    # The gainmatrices should be stored in the format they
    # are normally produced by the gaincalc module, namely:
    # {case}_{scenario}_{method}_maxweight_array_box{boxindex}.csv
    # where method refers to the method used to calculate the gain
    gainmatrices = []

    boxcount = 0

    for index, file in enumerate(os.listdir(noderankdata.gainloc)):
        if fnmatch.fnmatch(file, countlocation):
            boxcount += 1

    for boxindex in range(boxcount):
        gainmatrix = data_processing.read_gainmatrix(
            os.path.join(noderankdata.gainloc,
                         gainmatrix_filename.format(case, scenario,
                                                    method,
                                                    boxindex+1)))

        gainmatrices.append(gainmatrix)

    return gainmatrices


def looprank(mode, case, writeoutput, preprocessing=False):
    """Ranks the nodes in a network based on gain matrices already generated.

    Preprocessing is experimental and should always be set to False at this
    stage.

    """
    noderankdata = NoderankData(mode, case)

    countfile_template = '{}_{}_{}_maxweight_array_box*.csv'
    gainmatrix_filename = '{}_{}_{}_maxweight_array_box{:03d}.csv'

    # Define export directories and filenames
    # Get the directory to save in
    savedir = \
        config_setup.ensure_existance(
            os.path.join(noderankdata.saveloc,
                         'noderank'), make=True)

    rankinglist_filename = \
        os.path.join(savedir, '{}_{}_{}_rankinglist_{}_box{:03d}.csv')

    modgainmatrix_template = \
        os.path.join(savedir, '{}_{}_{}_modgainmatrix_box{:03d}.csv')

    originalgainmatrix_template = \
        os.path.join(savedir, '{}_{}_{}_originalgainmatrix_box{:03d}.csv')

    graphfile_template = os.path.join(savedir,
                                      '{}_{}_{}_{}_graph_{}_{}_box{:03d}.gml')

    importances_template = os.path.join(
        savedir, '{}_{}_{}_{}_importances_{}_{}_box{:03d}.csv')

    graph_filename = os.path.join(savedir,
                                  "{}_{}_{}_{}_graph_dumsup_box{:03d}.gml")

#    transientdict_name = os.path.join(savedir,
#                                      '{}_{}_{}_{}_transientdict.json')
#
#    basevaldict_name = os.path.join(savedir,
#                                    '{}_{}_{}_{}_basevaldict.json')

    boxrankdict_name = os.path.join(savedir,
                                    '{}_{}_{}_{}_boxrankdict.json')

    for scenario in noderankdata.scenarios:
        logging.info("Running scenario {}".format(scenario))
        # Update scenario-specific fields of weightcalcdata object
        noderankdata.scenariodata(scenario)
        for rank_method in noderankdata.rank_methods:
            for weight_method in noderankdata.weight_methods:
                # Test whether the 'rankinglist_rankingmethod_box001' CSV file
                # already exists
                testlocation = \
                    rankinglist_filename.format(case, scenario, weight_method,
                                                rank_method, 1)
                # TODO: Implement method in filename down below as well
                if not os.path.exists(testlocation):
                    # Continue with execution
                    countlocation = \
                        countfile_template.format(case, scenario,
                                                  weight_method)

                    gainmatrices = get_gainmatrices(noderankdata,
                                                    countlocation,
                                                    gainmatrix_filename, case,
                                                    scenario, weight_method)

                    # Create lists to store the backward ranking list
                    # for each box and associated gainmatrix ranking result
                    backward_rankinglists = []
                    backward_rankingdicts = []

                    for index, gainmatrix in enumerate(gainmatrices):
                        if preprocessing:
                            modgainmatrix, _ = \
                                gainmatrix_preprocessing(gainmatrix)
                        else:
                            modgainmatrix = gainmatrix

                        _, dummyweight = \
                            gainmatrix_preprocessing(gainmatrix)

                        # This is where the actual ranking calculation happens
                        rankingdicts, rankinglists, connections, \
                            variables, gains = \
                            calc_gainrank(modgainmatrix, noderankdata,
                                          rank_method, dummyweight)

                        # The rest of the function deals with storing the
                        # results in the desired formats

                        backward_rankinglists.append(rankinglists[0])
                        backward_rankingdicts.append(rankingdicts[0])

                        if writeoutput:
                            # Save the ranking list for each box
                            writecsv_looprank(
                                rankinglist_filename.format(
                                    case, scenario, weight_method, rank_method,
                                    index+1),
                                rankinglists[0])

                            # Save the modified gainmatrix
                            writecsv_looprank(
                                modgainmatrix_template.format(
                                    case, scenario, weight_method, index+1),
                                modgainmatrix)

                            # Save the original gainmatrix
                            writecsv_looprank(
                                originalgainmatrix_template.format(
                                    case, scenario, weight_method, index+1),
                                gainmatrix)

                            # Export graph files with dummy variables included
                            # in backward rankings if available
                            directions = ['backward']

                            if noderankdata.dummies:
                                dummystatus = 'withdummies'
                            else:
                                dummystatus = 'nodummies'

                            for direction, rankinglist, rankingdict, \
                                connection, variable, gain in zip(
                                    directions, rankinglists, rankingdicts,
                                    connections, variables, gains):
                                idtuple = (case, scenario, weight_method,
                                           direction, rank_method,
                                           dummystatus, index+1)

                                # Save the ranking list to file
                                savename = importances_template.format(
                                    *idtuple)
                                writecsv_looprank(savename, rankinglist)

                                # Save the graphs to file
                                graph, _ = \
                                    create_importance_graph(
                                        variable, connection, connection, gain,
                                        rankingdict)
                                graph_filename = \
                                    graphfile_template.format(*idtuple)
                                nx.readwrite.write_gml(graph, graph_filename)

                            if noderankdata.dummies:
                                # Export backward ranking graphs
                                # without dummy variables visible

                                # Backward ranking graph
                                direction = directions[0]
                                rankingdict = rankingdicts[0]
                                connectionmatrix = \
                                    noderankdata.connectionmatrix.T
                                gainmatrix = gainmatrix.T
                                graph, _ = \
                                    create_importance_graph(
                                        noderankdata.variablelist,
                                        connectionmatrix,
                                        connectionmatrix,
                                        gainmatrix,
                                        rankingdict)

                                nx.readwrite.write_gml(
                                    graph,
                                    graph_filename.format(case, scenario,
                                                          weight_method,
                                                          rank_method,
                                                          direction, index+1))

                                # Calculate and export normalised ranking lists
                                # with dummy variables exluded from results
                                for direction, rankingdict \
                                    in zip(directions[1:],
                                           rankingdicts[1:]):
                                    normalised_rankinglist = \
                                        normalise_rankinglist(
                                            rankingdict,
                                            noderankdata.variablelist)
                                    writecsv_looprank(
                                        importances_template.format(
                                            case, scenario, weight_method,
                                            rank_method,
                                            direction, index+1),
                                        normalised_rankinglist)

                    # Get the transient and base value dictionaries
                    _, _, boxrankdict = \
                        calc_transient_importancediffs(
                            backward_rankingdicts, noderankdata.variablelist)

                    # Store dictonaries using JSON

    #                data_processing.write_dictionary(
    #                    transientdict_name.format(case, scenario,
    #                                              weight_method, direction),
    #                    transientdict)

    #                data_processing.write_dictionary(
    #                    basevaldict_name.format(case, scenario,
    #                                            weight_method, direction),
    #                    basevaldict)

                    data_processing.write_dictionary(
                        boxrankdict_name.format(case, scenario,
                                                weight_method, direction),
                        boxrankdict)

                else:
                    logging.info("The requested results are in existence")

    return None
