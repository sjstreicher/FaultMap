# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 07:04:15 2014

@author: s13071832
"""



# TODO: This function is a clone of the object method above
# and therefore redundant but used in the transient ranking algorithm.
# It will be incorporated as soon as it is high enough priority
def calc_partialcorr_gainmatrix(connectionmatrix, tags_tsdata, *dataset):
    """Calculates the local gains in terms of the partial (Pearson's)
    correlation between the variables.

    connectionmatrix is the adjacency matrix

    tags_tsdata contains the time series data for the tags with variables
    in colums and sampling instances in rows

    """
    if isinstance(tags_tsdata, np.ndarray):
        inputdata = tags_tsdata
    else:
        inputdata = np.array(h5py.File(tags_tsdata, 'r')[dataset])
#    print "Total number of data points: ", inputdata.size
    # Calculate correlation matrix
    correlationmatrix = np.corrcoef(inputdata.T)
    # Calculate partial correlation matrix
    p_matrix = np.linalg.inv(correlationmatrix)
    d = p_matrix.diagonal()
    partialcorrelationmatrix = \
        np.where(connectionmatrix, -p_matrix/np.abs(np.sqrt(np.outer(d, d))),
                 0)

    return correlationmatrix, partialcorrelationmatrix


def partialcorrcalc(mode, case, writeoutput):
    """Returns the partial correlation matrix.

    Does not support optimizing with respect to time delays.

    """
    weightcalcdata = WeightcalcData(mode, case)
    partialmatcalculator = PartialCorrWeightcalc(weightcalcdata)

    for scenario in weightcalcdata.scenarios:
        logging.info("Running scenario {}".format(scenario))
        # Update scenario-specific fields of weightcalcdata object
        weightcalcdata.scenariodata(scenario)

        partialcorrmat, connectionmatrix, variables = partialmatcalculator.\
            partialcorr_gainmatrix(weightcalcdata)

        if writeoutput:
            # Define export directories and filenames
            partialmatdir = config_setup.ensure_existance(os.path.join(
                weightcalcdata.saveloc, 'partialcorr'), make=True)
            filename_template = os.path.join(partialmatdir, '{}_{}_{}.csv')

            def filename(name):
                return filename_template.format(case, scenario, name)
            # Write arrays to file
            np.savetxt(filename('partialcorr_array'), partialcorrmat,
                       delimiter=',')

            np.savetxt(filename('connectionmatrix'), connectionmatrix,
                       delimiter=',')

            writecsv_weightcalc(filename('variables'), variables,
                                variables)
