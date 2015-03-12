# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:56:02 2015

@author: STREICSJ1
"""

import sys
import logging
import numpy as np
import csv
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool


def writecsv_weightcalc(filename, items, header):
    """CSV writer customized for use in weightcalc function."""

    with open(filename, 'wb') as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(items)


# Need to rewrite this to not make use of classes
def calc_weights_onepair(causevarindex,
                         weightcalcdata, weightcalculator,
                         box, startindex, size,
                         newconnectionmatrix,
                         datalines_directional, datalines_absolute,
                         filename, method, boxindex, sigstatus, headerline,
                         causevar,
                         datalines_sigthresh_directional,
                         datalines_sigthresh_absolute,
                         datalines_neutral,
                         datalines_sigthresh_neutral,
                         sig_filename,
                         weight_array, delay_array, datastore,
                         affectedvarindex):

    affectedvar = weightcalcdata.variables[affectedvarindex]

    logging.info("Analysing effect of: " + causevar + " on " +
                 affectedvar + " for box number: " +
                 str(boxindex + 1))

    if not(newconnectionmatrix[affectedvarindex,
                               causevarindex] == 0):
                    weightlist = []
                    directional_weightlist = []
                    absolute_weightlist = []
                    sigthreshlist = []
                    directional_sigthreshlist = []
                    absolute_sigthreshlist = []

                    for delay in weightcalcdata.sample_delays:
                        logging.info("Now testing delay: " + str(delay))

                        causevardata = \
                            (box[:, causevarindex]
                                [startindex:startindex+size])

                        affectedvardata = \
                            (box[:, affectedvarindex]
                                [startindex+delay:startindex+size+delay])

                        # This function might be causing troube as it is in
                        # a class - rather import it directly
                        # How does it make use of weightcalcdata?
                        weight = weightcalculator.calcweight(causevardata,
                                                             affectedvardata,
                                                             weightcalcdata,
                                                             causevarindex,
                                                             affectedvarindex)

                        # Calculate significance thresholds at each data point
                        if weightcalcdata.allthresh:
                            sigthreshold = \
                                weightcalculator.calcsigthresh(affectedvardata,
                                                               causevardata)

                        if len(weight) > 1:
                            # If weight contains directional as well as
                            # absolute weights, write to separate lists
                            directional_weightlist.append(weight[0])
                            absolute_weightlist.append(weight[1])
                            # Same approach with significance thresholds
                            if weightcalcdata.allthresh:
                                directional_sigthreshlist.append(
                                    sigthreshold[0])
                                absolute_sigthreshlist.append(
                                    sigthreshold[1])

                        else:
                            weightlist.append(weight[0])
                            if weightcalcdata.allthresh:
                                sigthreshlist.append(sigthreshold[0])

                    directional_name = 'weights_directional'
                    absolute_name = 'weights_absolute'
                    neutral_name = 'weights'
                    # Provide names for the significance threshold file types
                    if weightcalcdata.allthresh:
                        sig_directional_name = 'sigthresh_directional'
                        sig_absolute_name = 'sigthresh_absolute'
                        sig_neutral_name = 'sigthresh'

                    if len(weight) > 1:
                        weightlist = [directional_weightlist,
                                      absolute_weightlist]

                        # Combine weight data
                        weights_thisvar_directional = np.asarray(weightlist[0])
                        weights_thisvar_directional = \
                            weights_thisvar_directional[:, np.newaxis]

                        weights_thisvar_absolute = np.asarray(weightlist[1])
                        weights_thisvar_absolute = \
                            weights_thisvar_absolute[:, np.newaxis]

                        datalines_directional = \
                            np.concatenate((datalines_directional,
                                            weights_thisvar_directional),
                                           axis=1)

                        datalines_absolute = \
                            np.concatenate((datalines_absolute,
                                            weights_thisvar_absolute),
                                           axis=1)

                        writecsv_weightcalc(filename(
                            directional_name,
                            method, boxindex+1, sigstatus, causevar),
                            datalines_directional, headerline)

                        writecsv_weightcalc(filename(
                            absolute_name,
                            method, boxindex+1, sigstatus, causevar),
                            datalines_absolute, headerline)

                        # Do the same for the significance threshold
                        if weightcalcdata.allthresh:
                            sigthreshlist = [directional_sigthreshlist,
                                             absolute_sigthreshlist]

                            sigthresh_thisvar_directional = \
                                np.asarray(sigthreshlist[0])
                            sigthresh_thisvar_directional = \
                                sigthresh_thisvar_directional[:, np.newaxis]

                            sigthresh_thisvar_absolute = \
                                np.asarray(sigthreshlist[1])
                            sigthresh_thisvar_absolute = \
                                sigthresh_thisvar_absolute[:, np.newaxis]

                            datalines_sigthresh_directional = np.concatenate(
                                (datalines_sigthresh_directional,
                                 sigthresh_thisvar_directional),
                                axis=1)

                            datalines_sigthresh_absolute = \
                                np.concatenate((datalines_sigthresh_absolute,
                                                sigthresh_thisvar_absolute),
                                               axis=1)

                            writecsv_weightcalc(sig_filename(
                                sig_directional_name,
                                method, boxindex+1, causevar),
                                datalines_sigthresh_directional, headerline)

                            writecsv_weightcalc(sig_filename(
                                sig_absolute_name,
                                method, boxindex+1, causevar),
                                datalines_sigthresh_absolute, headerline)

                    else:
                        weights_thisvar_neutral = np.asarray(weightlist)
                        weights_thisvar_neutral = \
                            weights_thisvar_neutral[:, np.newaxis]

                        datalines_neutral = \
                            np.concatenate((datalines_neutral,
                                            weights_thisvar_neutral), axis=1)

                        writecsv_weightcalc(filename(
                            neutral_name,
                            method, boxindex+1, sigstatus, causevar),
                            datalines_neutral, headerline)

                        # Write the significance thresholds to file
                        if weightcalcdata.allthresh:
                            sigthresh_thisvar_neutral = np.asarray(weightlist)
                            sigthresh_thisvar_neutral = \
                                sigthresh_thisvar_neutral[:, np.newaxis]

                            datalines_sigthresh_neutral = \
                                np.concatenate((datalines_sigthresh_neutral,
                                                sigthresh_thisvar_neutral),
                                               axis=1)

                            writecsv_weightcalc(sig_filename(
                                sig_neutral_name,
                                method, boxindex+1, causevar),
                                datalines_sigthresh_neutral, headerline)

                    # Generate and store report files according to each method
                    [weight_array, delay_array, datastore] = \
                        weightcalculator.report(weightcalcdata, causevarindex,
                                                affectedvarindex, weightlist,
                                                weight_array, delay_array,
                                                datastore)

    return weight_array, delay_array, datastore


if __name__ == '__main__':
    partial_gaincalc_onepair = partial(
                calc_weights_onepair,
                causevarindex,
                weightcalcdata, weightcalculator,
                box, startindex, size,
                newconnectionmatrix,
                datalines_directional, datalines_absolute,
                filename, method, boxindex, sigstatus, headerline,
                causevar,
                datalines_sigthresh_directional,
                datalines_sigthresh_absolute,
                datalines_neutral,
                datalines_sigthresh_neutral,
                sig_filename,
                weight_array, delay_array, datastore)

#    for affectedvarindex in weightcalcdata.affectedvarindexes:
#        weight_array, delay_array, datastore = \
#                    partial_gaincalc_onepair(affectedvarindex)

    pool = Pool(processes=4)
    result = pool.map(partial_gaincalc_onepair,
                      weightcalcdata.affectedvarindexes)

    # Can make use of CSV files or stdout
