# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 09:41:14 2014

@author: s13071832
"""

def calc_topedge_rank(gainmatrix, variables, m, topedgenum=10):
    """Calculates the ranking based on the top edges only.

    topedgenum is the number of largest edges to include in the ranking problem
    """

    # Identify the largest topedgenum elements in the gainmatrix
    largest_indexes = []

    # TODO: Implement changes

    return calc_simple_rank(gainmatrix, variables, m)



def calc_maingainrank(gainmatrix, noderankdata, dummycreation, dummyweight,
                      m):
    """Calculates the backward ranking for a truncated gainmatrix with only the
    most significant edges retained.

    """

    mainconnection, maingain, mainvariablelist = \
        data_processing.rankbackward(noderankdata.variablelist, gainmatrix,
                                     noderankdata.connectionmatrix,
                                     dummyweight, dummycreation)

    mainrankingdict, mainrankinglist = \
        calc_topedge_rank(maingain, mainvariablelist, m)

    return mainrankingdict, mainrankinglist, mainconnection, \
        mainvariablelist, maingain

        # include in looprank_static below similar looking code
                    mainrankingdict, mainrankinglist, mainconnection, \
                mainvariables, maingains = \
                calc_maingainrank(modgainmatrix, noderankdata, dummycreation,
                                  dummyweight, m)



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
                data_processing.read_connectionmatrix(connectionloc)

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
        boxes = data_processing.split_tsdata(tags_tsdata, dataset, samplerate,
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




                    gainmatrices = []

self.gainmatrix = data_processing.read_gainmatrix(gainloc)
            for gainmatrix in gainmatrices:



################################
# Testing reading gainmatrices
###############################



            # Count the number of gainmatrices that conform to the specific
            # case and scenario combination in the export files
            #


            # Modify the gainmatrix to have a specific mean
            # Should only be used for development analysis - generally destroys
            # information.
            # Not sure what effect will be if data is variance scaled as well
            if preprocessing:
                modgainmatrix, _ = \
                    gainmatrix_preprocessing(noderankdata.gainmatrix)
            else:
                modgainmatrix = noderankdata.gainmatrix

            _, dummyweight = \
                gainmatrix_preprocessing(noderankdata.gainmatrix)

            rankingdicts, rankinglists, connections, variables, gains = \
                calc_gainrank(modgainmatrix, noderankdata,
                              dummycreation,
                              alpha, dummyweight, m)

            if writeoutput:
                # Save the modified gainmatrix
                modgainmatrix_template = \
                    os.path.join(savedir, '{}_modgainmatrix.csv')
                savename = modgainmatrix_template.format(scenario)
                writecsv_looprank(savename, modgainmatrix)
                csvfile_template = os.path.join(savedir,
                                                '{}_{}_importances_{}.csv')

                # Save the original gainmatrix
                savename = originalgainmatrix_template.format(scenario,
                                                              boxindex+1)
                writecsv_looprank(savename, noderankdata.gainmatrix)

                # Export graph files with dummy variables included in
                # forward and backward rankings if available
                directions = ['blended', 'forward', 'backward']

                if dummycreation:
                    dummystatus = 'withdummies'
                else:
                    dummystatus = 'nodummies'

                # TODO: Do the same for meanchange

                graphfile_template = os.path.join(savedir,
                                                  '{}_{}_graph_{}.gml')

                for direction, rankinglist, rankingdict, connection, \
                    variable, gain in zip(directions, rankinglists,
                                          rankingdicts,
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
                    graph, _ = \
                        create_importance_graph(noderankdata.variablelist,
                                                noderankdata.connectionmatrix,
                                                noderankdata.connectionmatrix,
                                                noderankdata.gainmatrix,
                                                rankingdict)
                    graph_filename = os.path.join(noderankdata.saveloc,
                                                  'noderank',
                                                  "{}_{}_graph_dumsup.gml"
                                                  .format(scenario, direction))

                    nx.readwrite.write_gml(graph, graph_filename)

                    # Backward ranking graph
                    direction = directions[2]
                    rankingdict = rankingdicts[2]
                    connectionmatrix = noderankdata.connectionmatrix.T
                    gainmatrix = noderankdata.gainmatrix.T
                    graph, _ = \
                        create_importance_graph(noderankdata.variablelist,
                                                connectionmatrix,
                                                connectionmatrix,
                                                gainmatrix,
                                                rankingdict)
                    graph_filename = os.path.join(noderankdata.saveloc,
                                                  'noderank',
                                                  "{}_{}_graph_dumsup.gml"
                                                  .format(scenario, direction))

                    nx.readwrite.write_gml(graph, graph_filename)

                    # Calculate and export normalised ranking lists
                    # with dummy variables exluded from results
                    for direction, rankingdict in zip(directions[1:],
                                                      rankingdicts[1:]):
                        normalised_rankinglist = \
                            normalise_rankinglist(rankingdict,
                                                  noderankdata.variablelist)

                        savename = os.path.join(noderankdata.saveloc,
                                                'noderank',
                                                '{}_{}_importances_dumsup.csv'
                                                .format(scenario, direction))
                        writecsv_looprank(savename, normalised_rankinglist)
        else:
            logging.info("The requested results are in existence")

    logging.info("Done with static ranking")