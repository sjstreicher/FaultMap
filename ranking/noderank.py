"""Ranks nodes in a network when provided with a connection and gainmatrix.

@author St. Elmo Wilken, Simon Streicher

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
import matplotlib.pyplot as plt

# Own libraries
import formatmatrices
import config_setup
import ranking

import networkgen


def writecsv_looprank(filename, items):
    with open(filename, 'wb') as f:
        csv.writer(f).writerows(items)


def calc_simple_rank(gainmatrix, variables, m):
    """Constructs the ranking dictionary using the eigenvector approach
    i.e. Ax = x where A is the local gain matrix.

    m is the weight of the full connectivity matrix used to ensure
    graph is not sub-stochastic

    """
    # Length of gain matrix = number of nodes
    gainmatrix = np.asarray(gainmatrix)
    n = len(gainmatrix)
    s_matrix = (1.0 / n) * np.ones((n, n))
    # Basic PageRank algorithm
    m_matrix = (1 - m) * gainmatrix + m * s_matrix
    # Calculate eigenvalues and eigenvectors as usual
    [eigval, eigvec] = np.linalg.eig(m_matrix)
    maxeigindex = np.argmax(eigval)
    # Store value for downstream checking
    # TODO: Downstream checking not implemented yet
#    maxeig = eigval[maxeigindex].real
    # Cuts array into the eigenvector corrosponding to the eigenvalue above
    rankarray = eigvec[:, maxeigindex]
    # This is the 1-dimensional array composed of rankings (normalised)
    rankarray = (1 / sum(rankarray)) * rankarray
    # Remove the useless imaginary +0j
    rankarray = rankarray.real

    # Create a dictionary of the rankings with their respective nodes
    # i.e. {NODE:RANKING}
    rankingdict = dict(zip(variables, rankarray))

    rankinglist = sorted(rankingdict.iteritems(), key=operator.itemgetter(1),
                         reverse=True)

    return rankingdict, rankinglist


def calc_blended_rank(forwardrank, backwardrank, variablelist,
                      alpha):
    """This method creates a blended ranking profile."""
    rankingdict = dict()
    for variable in variablelist:
        rankingdict[variable] = abs(((1 - alpha) * forwardrank[variable] +
                                     (alpha) * backwardrank[variable]))

    total = sum(rankingdict.values())
    # Normalise rankings
    for variable in variablelist:
        rankingdict[variable] = rankingdict[variable] / total

    rankinglist = sorted(rankingdict.iteritems(), key=operator.itemgetter(1),
                         reverse=True)

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
    """Creates dictionary with a vector of successive differences in importance
    scores between boxes for each variable entry.

    """
    transientdict = dict()
    basevaldict = dict()
    for variable in variablelist:
        diffvect = np.empty((1, len(rankingdicts)-1))[0]
        diffvect[:] = np.NAN
        basevaldict[variable] = rankingdicts[0][variable]
        # Get initial previous importance
        prev_importance = basevaldict[variable]
        for index, rankingdict in enumerate(rankingdicts[1:]):
            diffvect[index] = rankingdict[variable] - prev_importance
            prev_importance = rankingdict[variable]
        transientdict[variable] = diffvect

    return transientdict, basevaldict


def plot_transient_importances(variables, transientdict, basevaldict):
    """Plots the transient importance for the specified variables.
    Plots both absolute rankings over time as well as ranking differences only.

    """
    transient_val_no = len(transientdict[variables[1]])
    # Transient rankings down in rows, each variable contained in a column
    diffplot = np.zeros((transient_val_no+1, len(variables)))
    absplot = np.zeros_like(diffplot)

    for index, variable in enumerate(variables):
        diffplot[:, index][1:] = transientdict[variable]
        absplot[0, index] = basevaldict[variable]
        absplot[:, index][1:] = diffplot[:, index][1:] + basevaldict[variable]

    bins = range(transient_val_no+1)

    plt.figure(1)
    plt.plot(bins, diffplot)
    plt.title('Relative importance variations over time')

    plt.figure(2)
    plt.plot(bins, absplot)
    plt.title('Absolute importance scores over time')

    return plt.figure(1), plt.figure(2)


def create_importance_graph(variablelist, closedconnections,
                            openconnections, gainmatrix, ranking):
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
        closedgraph.add_node(node, importance=ranking[node])

    return closedgraph, opengraph


def calc_gainmatrix(connectionmatrix, tags_tsdata, dataset,
                    method='partial_correlation'):
    """Calculates the required gainmatrix from tags time series data.

    Can make use of either the partial correlation or transfer entropy method
    for determining weights.

    """

    # TODO: Rewrite to use weightcalc

    if method == 'partial_correlation':
        # Get the partialcorr gainmatrix
        # TODO: Implement partial correlation weight calculation
        gainmatrix = None

    elif method == 'transfer_entropy':
        # TODO: Implement transfer entropy weight calculation
        gainmatrix = None

    return gainmatrix


def gainmatrix_preprocessing(gainmatrix, connectionmatrix):
    """Moves the mean and scales the variance of the elements in the
    gainmatrix to a specified value.

    Only operates on nonzero weights.

    """
    # Get the mean of the samples in the gainmatrix that correspond
    # to the desired connectionmatrix.
    counter = 0
    gainsum = 0
    for col, row in itertools.izip(connectionmatrix.nonzero()[0],
                                   connectionmatrix.nonzero()[1]):
        gainsum += gainmatrix[col, row]
        counter += 1

    currentmean = gainsum / counter
    meanscale = 1. / currentmean

    # Write meandiff to all gainmatrix elements indicated by connectionmatrix
    modgainmatrix = np.zeros_like(gainmatrix)

    for col, row in itertools.izip(connectionmatrix.nonzero()[0],
                                   connectionmatrix.nonzero()[1]):
        modgainmatrix[col, row] = gainmatrix[col, row] * meanscale

    return modgainmatrix, currentmean


def calc_gainrank(gainmatrix, variables, connectionmatrix, dummycreation,
                  alpha, dummyweight, m):
    """Calculates the forward and backward rankings.

    """

    forwardconnection, forwardgain, forwardvariablelist = \
        formatmatrices.rankforward(variables, gainmatrix, connectionmatrix,
                                   dummyweight, dummycreation)

    backwardconnection, backwardgain, backwardvariablelist = \
        formatmatrices.rankbackward(variables, gainmatrix, connectionmatrix,
                                    dummyweight, dummycreation)

    forwardrankingdict, forwardrankinglist = \
        calc_simple_rank(forwardgain, forwardvariablelist, m)

    backwardrankingdict, backwardrankinglist = \
        calc_simple_rank(backwardgain, backwardvariablelist, m)

    blendedrankingdict, blendedrankinglist = \
        calc_blended_rank(forwardrankingdict, backwardrankingdict,
                          variables, alpha)

    rankingdicts = [blendedrankingdict, forwardrankingdict,
                    backwardrankingdict]
    rankinglists = [blendedrankinglist, forwardrankinglist,
                    backwardrankinglist]
    connections = [connectionmatrix, forwardconnection, backwardconnection]
    variables = [variables, forwardvariablelist, backwardvariablelist]
    gains = [gainmatrix, np.array(forwardgain), np.array(backwardgain)]

    return rankingdicts, rankinglists, connections, variables, gains


def looprank_static(mode, case, dummycreation, writeoutput,
                    alpha, m):
    """Ranks the nodes in a network based on a single gain matrix calculation.

    For calculation of rank over time, see looprank_transient.

    """

    # Only to be used in rare development test cases
    preprocessing = False

    saveloc, casedir, infodynamicsloc = config_setup.runsetup(mode, case)

    # Load case config file
    caseconfig = json.load(open(os.path.join(casedir, case + '_noderank' +
                                             '.json')))
    # Get scenarios
    scenarios = caseconfig['scenarios']
    # Get data type
    datatype = caseconfig['datatype']

    for scenario in scenarios:
        logging.info("Running scenario {}".format(scenario))
        if datatype == 'file':

            # Get time series data
#            tags_tsdata = os.path.join(casedir, 'data',
#                                       caseconfig[scenario]['data'])
            # Get connection (adjacency) matrix
            connectionloc = os.path.join(casedir, 'connections',
                                         caseconfig[scenario]['connections'])
            # Get dataset name
#            dataset = caseconfig[scenario]['dataset']
            # Get the variables and connection matrix
            [variablelist, connectionmatrix] = \
                formatmatrices.read_connectionmatrix(connectionloc)

            # Calculate the gainmatrix
            # TODO: Use result from weightcalc as gainmatrix
            gainloc = os.path.join(casedir, 'gainmatrix',
                                   caseconfig[scenario]['gainmatrix'])

            gainmatrix = formatmatrices.read_gainmatrix(gainloc)
#            gainmatrix = calc_gainmatrix(connectionmatrix,
#                                         tags_tsdata, dataset)

        elif datatype == 'function':
            # Get variables, connection matrix and gainmatrix
            network_gen = caseconfig[scenario]['networkgen']
            connectionmatrix, gainmatrix, variablelist, _ = eval('networkgen.' + network_gen)()

        logging.info("Number of tags: {}".format(len(variablelist)))

        # Modify the gainmatrix to have a specific mean
        # Should only be used for development analysis - generally destroys
        # information.
        # Not sure what effect will be if data is variance scaled as well
        if preprocessing:
            modgainmatrix, _ = \
                gainmatrix_preprocessing(gainmatrix, connectionmatrix)
        else:
            modgainmatrix = gainmatrix

        _, dummyweight = gainmatrix_preprocessing(gainmatrix, connectionmatrix)

        rankingdicts, rankinglists, connections, variables, gains = \
            calc_gainrank(modgainmatrix, variablelist, connectionmatrix,
                          dummycreation,
                          alpha, dummyweight, m)

        if writeoutput:
            # Get the directory to save in
            savedir = \
                config_setup.ensure_existance(os.path.join(saveloc,
                                                           'noderank'),
                                              make=True)
            # Save the modified gainmatrix
            modgainmatrix_template = \
                os.path.join(savedir, '{}_modgainmatrix.csv')
            savename = modgainmatrix_template.format(scenario)
            writecsv_looprank(savename, modgainmatrix)

            # Save the original gainmatrix
            originalgainmatrix_template = \
                os.path.join(savedir, '{}_originalgainmatrix.csv')
            savename = originalgainmatrix_template.format(scenario)
            writecsv_looprank(savename, gainmatrix)

            # Export graph files with dummy variables included in
            # forward and backward rankings if available
            directions = ['blended', 'forward', 'backward']

            if dummycreation:
                dummystatus = 'withdummies'
            else:
                dummystatus = 'nodummies'

            # TODO: Do the same for meanchange

            csvfile_template = os.path.join(savedir,
                                            '{}_{}_importances_{}.csv')
            graphfile_template = os.path.join(savedir, '{}_{}_graph_{}.gml')

            for direction, rankinglist, rankingdict, connection, \
                variable, gain in zip(directions, rankinglists, rankingdicts,
                                      connections, variables, gains):
                idtuple = (scenario, direction, dummystatus)
                # Save the ranking list to file
                savename = csvfile_template.format(*idtuple)
                writecsv_looprank(savename, rankinglist)
                # Save the graphs to file
                graph, _ = create_importance_graph(variable, connection,
                                                   connection, gain,
                                                   rankingdict)
                graph_filename = graphfile_template.format(*idtuple)

                nx.readwrite.write_gml(graph, graph_filename)

            if dummycreation:

                # Export forward and backward ranking graphs
                # without dummy variables visible

                # Forward ranking graph
                direction = directions[1]
                rankingdict = rankingdicts[1]
                graph, _ = create_importance_graph(variablelist,
                                                   connectionmatrix,
                                                   connectionmatrix,
                                                   gainmatrix,
                                                   rankingdict)
                graph_filename = os.path.join(saveloc, 'noderank',
                                              "{}_{}_graph_dumsup.gml"
                                              .format(scenario, direction))

                nx.readwrite.write_gml(graph, graph_filename)

                # Backward ranking graph
                direction = directions[2]
                rankingdict = rankingdicts[2]
                connectionmatrix = connectionmatrix.T
                gainmatrix = gainmatrix.T
                graph, _ = create_importance_graph(variablelist,
                                                   connectionmatrix,
                                                   connectionmatrix,
                                                   gainmatrix,
                                                   rankingdict)
                graph_filename = os.path.join(saveloc, 'noderank',
                                              "{}_{}_graph_dumsup.gml"
                                              .format(scenario, direction))

                nx.readwrite.write_gml(graph, graph_filename)

                # Calculate and export normalised ranking lists
                # with dummy variables exluded from results
                for direction, rankingdict in zip(directions[1:],
                                                  rankingdicts[1:]):
                    normalised_rankinglist = \
                        normalise_rankinglist(rankingdict, variablelist)

                    savename = os.path.join(saveloc, 'noderank',
                                            '{}_{}_importances_dumsup.csv'
                                            .format(scenario, direction))
                    writecsv_looprank(savename, normalised_rankinglist)

    logging.info("Done with static ranking")

    return None


def looprank_transient(mode, case, dummycreation, writeoutput,
                       plotting=False):
    """Ranks the nodes in a network based over time.

    """

    # Note: This is still a work in progress
    # TODO: Rewrite to make use of multiple calls of looprank_static

    saveloc, casedir, infodynamicsloc = config_setup.runsetup(mode, case)

    # Load case config file
    caseconfig = json.load(open(os.path.join(casedir, case + '.json')))
    # Get scenarios
    scenarios = caseconfig['scenarios']
    # Get data type
    datatype = caseconfig['datatype']
    # Get sample rate
    samplerate = caseconfig['sampling_rate']

    for scenario in scenarios:
        logging.info("Running scenario {}".format(scenario))
        if datatype == 'file':

             # Get time series data
            tags_tsdata = os.path.join(casedir, 'data',
                                       caseconfig[scenario]['data'])
            # Get connection (adjacency) matrix
            connectionloc = os.path.join(casedir, 'connections',
                                         caseconfig[scenario]['connections'])
            # Get dataset name
            dataset = caseconfig[scenario]['dataset']
            # Get the variables and connection matrix
            [variablelist, connectionmatrix] = \
                formatmatrices.read_connectionmatrix(connectionloc)

            # Calculate the gainmatrix
            gainmatrix = calc_gainmatrix(connectionmatrix,
                                         tags_tsdata, dataset)
            if writeoutput:
            # TODO: Refine name
                savename = os.path.join(saveloc, "gainmatrix.csv")
                np.savetxt(savename, gainmatrix, delimiter=',')

            boxnum = caseconfig['boxnum']
            boxsize = caseconfig['boxsize']

        elif datatype == 'function':
            # Get variables, connection matrix and gainmatrix
            network_gen = caseconfig[scenario]['networkgen']
            connectionmatrix, gainmatrix, variablelist, _ = eval(network_gen)()

        logging.info("Number of tags: {}".format(len(variablelist)))

        # Split the tags_tsdata into sets (boxes) useful for calculating
        # transient correlations
        boxes = formatmatrices.split_tsdata(tags_tsdata, dataset, samplerate,
                                            boxsize, boxnum)

        # Calculate gain matrix for each box
        gainmatrices = \
            [ranking.gaincalc.calc_partialcorr_gainmatrix(connectionmatrix,
                                                          box, dataset)[1]
             for box in boxes]

        rankinglists = []
        rankingdicts = []

        weightdir = \
            config_setup.ensure_existance(os.path.join(saveloc, 'weightcalc'),
                                          make=True)
        gain_template = os.path.join(weightdir, '{}_gainmatrix_{:03d}.csv')
        rank_template = os.path.join(saveloc, 'importances_{:03d}.csv')

        for index, gainmatrix in enumerate(gainmatrices):
            # Store the gainmatrix
            gain_filename = gain_template.format(scenario, index)
            np.savetxt(gain_filename, gainmatrix, delimiter=',')

            rankingdict, rankinglist, _, _, _ = \
                calc_gainrank(gainmatrix, variablelist, connectionmatrix)

            rankinglists.append(rankinglist[0])

            savename = rank_template.format(index)
            writecsv_looprank(savename, rankinglist[0])

            rankingdicts.append(rankingdict[0])

        transientdict, basevaldict = \
            calc_transient_importancediffs(rankingdicts, variablelist)

        # Plotting functions
        if plotting:
            diffplot, absplot = plot_transient_importances(variablelist,
                                                           transientdict,
                                                           basevaldict)
            diffplot_filename = os.path.join(saveloc,
                                             "{}_diffplot.pdf"
                                             .format(scenario))
            absplot_filename = os.path.join(saveloc,
                                            "{}_absplot.pdf"
                                            .format(scenario))
            diffplot.savefig(diffplot_filename)
            absplot.savefig(absplot_filename)

        logging.info("Done with transient rankings")

        return None
