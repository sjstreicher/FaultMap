# -*- coding: utf-8 -*-
"""This module stores the weight calculator classes
used by the gaincalc module.

"""
# Standard libraries
import numpy as np
import logging

# Non-standard external libraries
import pygeonetwork

# Own libraries
import transentropy


class PartialCorrWeightcalc:
    """This class provides methods for calculating the weights according to
    the partial correlation method.

    The option of handling delays in similar fashion to that of
    cross-correlation is provided. However, the ambiguity of using any delays
    in the partial correlation calculation should be noted.

    """

    def __init__(self, weightcalcdata):
        """Read the files or functions and returns required data fields.

        """

        # Use the same threshold values as used by Bauer for correlation.
        # TODO: Adjust this for the case of partial correlation
        self.threshcorr = (1.85*(weightcalcdata.testsize**(-0.41))) + \
            (2.37*(weightcalcdata.testsize**(-0.53)))
        self.threshdir = 0.46*(weightcalcdata.testsize**(-0.16))
#        logging.info("Directionality threshold: " + str(self.threshdir))
#        logging.info("Correlation threshold: " + str(self.threshcorr))

        self.connections_used = weightcalcdata.connections_used

        self.data_header = ['causevar', 'affectedvar', 'base_corr',
                            'max_corr', 'max_delay', 'max_index',
                            'signchange', 'corrthreshpass',
                            'dirrthreshpass', 'dirval']

    def calcweight(self, causevardata, affectedvardata, weightcalcdata,
                   causevarindex, affectedvarindex):
        """Calculates the partial correlation between two elements in the
        complete dataset.

        It is important to note that this function differs from the others
        in the sense that it requires the full dataset to be known.

        """

        startindex = weightcalcdata.startindex
        size = weightcalcdata.testsize
        vardims = len(weightcalcdata.variables)

        # Get inputdata and initial connectionmatrix
        calcdata = (weightcalcdata.inputdata[:, :]
                    [startindex:startindex+size])

        newvariables = weightcalcdata.variables

        if self.connections_used:
            newconnectionmatrix = weightcalcdata.connectionmatrix
        else:
            # If there is not connection matrix given, create a fully
            # connected connection matrix
            newconnectionmatrix = np.ones((vardims, vardims))

        # Delete all columns not listed in causevarindexes
        # Calculation is cheap, and in order to keep things simple the
        # causevarindexes are the sole means of doing selective calculation
        dellist = []
        for index in range(vardims):
            if index not in weightcalcdata.causevarindexes:
                dellist.append(index)
                logging.info("Deleted column " + str(index))

        # Delete all columns listed in dellist from calcdata
        newcalcdata = np.delete(calcdata, dellist, 1)

        # Delete all indexes listed in dellist from variables
        newvariables = np.delete(newvariables, dellist)

        # Delete all rows and columns listed in dellist
        # from connectionmatrix
        newconnectionmatrix = np.delete(newconnectionmatrix, dellist, 1)
        newconnectionmatrix = np.delete(newconnectionmatrix, dellist, 0)

        # Calculate correlation matrix
        correlationmatrix = np.corrcoef(newcalcdata.T)
        # Calculate partial correlation matrix
        p_matrix = np.linalg.inv(correlationmatrix)
        d = p_matrix.diagonal()
        partialcorrelationmatrix = \
            np.where(newconnectionmatrix,
                     -p_matrix/np.abs(np.sqrt(np.outer(d, d))), 0)

        return partialcorrelationmatrix[affectedvarindex, causevarindex]

    def report(self, weightcalcdata, causevarindex, affectedvarindex,
               weightlist, weight_array, delay_array, datastore, sigtest):
        """Calculates and reports the relevant output for each combination
        of variables tested.

        """
        variables = weightcalcdata.variables
        causevar = variables[causevarindex]
        affectedvar = variables[affectedvarindex]

        maxval = max(weightlist)
        minval = min(weightlist)
        # Value used to break tie between maxval and minval if 1 and -1
        tol = 0.
        # Always select maxval if both are equal
        # This is to ensure that 1 is selected above -1
        if (maxval + (minval + tol)) >= 0:
            maxcorr = maxval
        else:
            maxcorr = minval

        delay_index = weightlist.index(maxcorr)

        # Correlation thresholds from Bauer2008 eq. 4
        maxcorr_abs = max(maxval, abs(minval))
        bestdelay = weightcalcdata.actual_delays[delay_index]
        directionindex = 2 * (abs(maxval + minval) /
                              (maxval + abs(minval)))

        signchange = not ((weightlist[0] / weightlist[delay_index]) >= 0)
        corrthreshpass = (maxcorr_abs >= self.threshcorr)
        dirthreshpass = (directionindex >= self.threshdir)

        logging.info("Maximum correlation value: " + str(maxval))
        logging.info("Minimum correlation value: " + str(minval))
        logging.info("The maximum correlation between " + causevar +
                     " and " + affectedvar + " is: " + str(maxcorr))
        logging.info("The corresponding delay is: " +
                     str(bestdelay))
        logging.info("The correlation with no delay is: "
                     + str(weightlist[0]))
        logging.info("Correlation threshold passed: " +
                     str(corrthreshpass))
        logging.info("Directionality value: " + str(directionindex))
        logging.info("Directionality threshold passed: " +
                     str(dirthreshpass))

        corrthreshpass = None
        dirthreshpass = None
        if sigtest:
            corrthreshpass = (maxcorr_abs >= self.threshcorr)
            dirthreshpass = (directionindex >= self.threshdir)
            if not (corrthreshpass and dirthreshpass):
                maxcorr = 0

        weight_array[affectedvarindex, causevarindex] = maxcorr

#        # Replace all nan by zero
#        nanlocs = np.isnan(weight_array)
#        weight_array[nanlocs] = 0

        delay_array[affectedvarindex, causevarindex] = bestdelay

        dataline = [causevar, affectedvar, str(weightlist[0]),
                    maxcorr, str(bestdelay), str(delay_index),
                    signchange, corrthreshpass, dirthreshpass, directionindex]

        datastore.append(dataline)

        return weight_array, delay_array, datastore


class CorrWeightcalc:
    """This class provides methods for calculating the weights according to the
    cross-correlation method.

    """
    def __init__(self, weightcalcdata):
        """Read the files or functions and returns required data fields.

        """
        self.threshcorr = (1.85*(weightcalcdata.testsize**(-0.41))) + \
            (2.37*(weightcalcdata.testsize**(-0.53)))
        self.threshdir = 0.46*(weightcalcdata.testsize**(-0.16))
#        logging.info("Directionality threshold: " + str(self.threshdir))
#        logging.info("Correlation threshold: " + str(self.threshcorr))

        self.data_header = ['causevar', 'affectedvar', 'base_corr',
                            'max_corr', 'max_delay', 'max_index',
                            'signchange', 'corrthreshpass',
                            'dirrthreshpass', 'dirval']

    def calcweight(self, causevardata, affectedvardata, weightcalcdata,
                   causevarindex, affectedvarindex):
        """Calculates the correlation between two vectors containing
        timer series data.

        """
        corrval = np.corrcoef(causevardata.T, affectedvardata.T)[1, 0]
        return [corrval]

    def report(self, weightcalcdata, causevarindex, affectedvarindex,
               weightlist, weight_array, delay_array, datastore, sigtest):
        """Calculates and reports the relevant output for each combination
        of variables tested.

        """
        variables = weightcalcdata.variables
        causevar = variables[causevarindex]
        affectedvar = variables[affectedvarindex]

        maxval = max(weightlist)
        minval = min(weightlist)
        # Value used to break tie between maxval and minval if 1 and -1
        tol = 0.
        # Always select maxval if both are equal
        # This is to ensure that 1 is selected above -1
        if (maxval + (minval + tol)) >= 0:
            maxcorr = maxval
        else:
            maxcorr = minval

        delay_index = weightlist.index(maxcorr)

        # Correlation thresholds from Bauer2008 eq. 4
        maxcorr_abs = max(maxval, abs(minval))
        bestdelay = weightcalcdata.actual_delays[delay_index]
        directionindex = 2 * (abs(maxval + minval) /
                              (maxval + abs(minval)))

        signchange = not ((weightlist[0] / weightlist[delay_index]) >= 0)
        corrthreshpass = (maxcorr_abs >= self.threshcorr)
        dirthreshpass = (directionindex >= self.threshdir)

        logging.info("Maximum correlation value: " + str(maxval))
        logging.info("Minimum correlation value: " + str(minval))
        logging.info("The maximum correlation between " + causevar +
                     " and " + affectedvar + " is: " + str(maxcorr))
        logging.info("The corresponding delay is: " +
                     str(bestdelay))
        logging.info("The correlation with no delay is: "
                     + str(weightlist[0]))
        logging.info("Correlation threshold passed: " +
                     str(corrthreshpass))
        logging.info("Directionality value: " + str(directionindex))
        logging.info("Directionality threshold passed: " +
                     str(dirthreshpass))

        corrthreshpass = None
        dirthreshpass = None
        if sigtest:
            corrthreshpass = (maxcorr_abs >= self.threshcorr)
            dirthreshpass = (directionindex >= self.threshdir)
            if not (corrthreshpass and dirthreshpass):
                maxcorr = 0

        weight_array[affectedvarindex, causevarindex] = maxcorr

#        # Replace all nan by zero
#        nanlocs = np.isnan(weight_array)
#        weight_array[nanlocs] = 0

        delay_array[affectedvarindex, causevarindex] = bestdelay

        dataline = [causevar, affectedvar, str(weightlist[0]),
                    maxcorr, str(bestdelay), str(delay_index),
                    signchange, corrthreshpass, dirthreshpass, directionindex]

        datastore.append(dataline)

        return weight_array, delay_array, datastore


class TransentWeightcalc:
    """This class provides methods for calculating the weights according to
    the transfer entropy method.

    """

    def __init__(self, weightcalcdata):
        self.data_header = ['causevar', 'affectedvar', 'base_ent',
                            'max_ent', 'max_delay', 'max_index', 'threshpass']
        # Setup Java class for infodynamics toolkit
        self.teCalc = \
            transentropy.setup_infodynamics_te(weightcalcdata.normalize)

    def calcweight(self, causevardata, affectedvardata, weightcalcdata,
                   causevarindex, affectedvarindex):
        """"Calculates the transfer entropy between two vectors containing
        timer series data.

        """
        # Calculate transfer entropy as the difference
        # between the forward and backwards entropy
        transent_fwd = \
            transentropy.calc_infodynamics_te(self.teCalc, affectedvardata.T,
                                              causevardata.T)
        transent_bwd = \
            transentropy.calc_infodynamics_te(self.teCalc, causevardata.T,
                                              affectedvardata.T)

        transent_directional = transent_fwd - transent_bwd
        transent_absolute = transent_fwd

        # Do not pass negatives on to weight array
#        if transent_directional < 0:
#            transent_directional = 0
#
#        if transent_absolute < 0:
#            transent_absolute = 0

        return [transent_directional, transent_absolute]

    def report(self, weightcalcdata, causevarindex, affectedvarindex,
               weightlist, weight_array, delay_array, datastore, sigtest,
               te_thresh_method='rankorder'):
        """Calculates and reports the relevant output for each combination
        of variables tested.

        """

        variables = weightcalcdata.variables
        causevar = variables[causevarindex]
        affectedvar = variables[affectedvarindex]
        inputdata = weightcalcdata.inputdata

        # We already know that when dealing with transfer entropy
        # the weightlist will consist of a list of lists
        weightlist_directional = weightlist[0]
#        weightlist_absolute = weightlist[1]

        size = weightcalcdata.testsize
        startindex = weightcalcdata.startindex

        # Do everything for the directional case

        maxval_directional = max(weightlist_directional)
        delay_index_directional = \
            weightlist_directional.index(maxval_directional)
        bestdelay_directional = \
            weightcalcdata.actual_delays[delay_index_directional]
        bestdelay_sample = \
            weightcalcdata.sample_delays[delay_index_directional]
        delay_array[affectedvarindex, causevarindex] = \
            bestdelay_directional

        logging.info("The maximum directional TE between " + causevar +
                     " and " + affectedvar + " is: " + str(maxval_directional))

        threshpass_directional = None
        if sigtest:
            # Calculate threshold for transfer entropy
            thresh_causevardata = \
                inputdata[:, causevarindex][startindex:startindex+size]
            thresh_affectedvardata = \
                inputdata[:, affectedvarindex][startindex + bestdelay_sample:
                                               startindex + size +
                                               bestdelay_sample]
            if te_thresh_method == 'rankorder':
                self.thresh_rankorder(thresh_affectedvardata.T,
                                      thresh_causevardata.T)

            elif te_thresh_method == 'sixsigma':
                self.thresh_sixsigma(thresh_affectedvardata.T,
                                     thresh_causevardata.T)

            logging.info("The directional TE threshold is: "
                         + str(self.threshent_directional))

            if maxval_directional >= self.threshent_directional:
                threshpass_directional = True
            else:
                threshpass_directional = False
                maxval_directional = 0

            logging.info("TE threshold passed: " + str(threshpass_directional))

        weight_array[affectedvarindex, causevarindex] = maxval_directional

        dataline = [causevar, affectedvar, str(weightlist_directional[0]),
                    maxval_directional, str(bestdelay_directional),
                    str(delay_index_directional),
                    threshpass_directional]

        datastore.append(dataline)

        logging.info("The corresponding delay is: "
                     + str(bestdelay_directional))
        logging.info("The TE with no delay is: " + str(weightlist[0][0]))

        return weight_array, delay_array, datastore

    def calc_surr_te(self, affected_data, causal_data, num):
        """Calculates surrogate transfer entropy values for significance
        threshold purposes.

        Generates surrogate time series data by making use of the iAAFT method
        (see Schreiber 2000a).

        Returns list of surrogate transfer entropy values of length num.

        """
        tsdata = np.array([affected_data, causal_data])
        # TODO: Research required number of iterations for good surrogate data
        surr_tsdata = \
            [pygeonetwork.surrogates.Surrogates.SmallTestData().
                get_refined_AAFT_surrogates(tsdata, 10)
             for n in range(num)]

        surr_te_fwd = [transentropy.calc_infodynamics_te(self.teCalc,
                                                         surr_tsdata[n][0],
                                                         surr_tsdata[n][1])
                       for n in range(num)]
        surr_te_bwd = [transentropy.calc_infodynamics_te(self.teCalc,
                                                         surr_tsdata[n][1],
                                                         surr_tsdata[n][0])
                       for n in range(num)]

        surr_te_directional = \
            [surr_te_fwd[n] - surr_te_bwd[n] for n in range(num)]

        surr_te_absolute = [surr_te_fwd[n] for n in range(num)]

        return surr_te_directional, surr_te_absolute

    def thresh_rankorder(self, affected_data, causal_data):
        """Calculates the minimum threshold required for a transfer entropy
        value to be considered significant.

        Makes use of a 95% single-sided certainty and a rank-order method.
        This correlates to taking the maximum transfer entropy from 19
        surrogate transfer entropy calculations as the threshold,
        see Schreiber2000a.

        """
        surr_te_directional, surr_te_absolute = \
            self.calc_surr_te(affected_data, causal_data, 19)

        self.threshent_directional = max(surr_te_directional)
        self.threshent_absolute = max(surr_te_absolute)

    def thresh_sixsigma(self, affected_data, causal_data):
        """Calculates the minimum threshold required for a transfer entropy
        value to be considered significant.

        Makes use of a six sigma Gaussian check as done in Bauer2005 with 30
        samples of surrogate data.

        """
        surr_te_directional, surr_te_absolute = \
            self.calc_surr_te(affected_data, causal_data, 30)

        surr_te_directional_mean = np.mean(surr_te_directional)
        surr_te_directional_stdev = np.std(surr_te_directional)

        surr_te_absolute_mean = np.mean(surr_te_absolute)
        surr_te_absolute_stdev = np.std(surr_te_absolute)

        self.threshent_directional = (6 * surr_te_directional_stdev) + \
            surr_te_directional_mean

        self.threshent_absolute = (6 * surr_te_absolute_stdev) + \
            surr_te_absolute_mean
