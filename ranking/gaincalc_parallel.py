# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 16:02:39 2015

@author: Simon Streicher

Calculates the gain (transfer entropy) for each causal and affected variable
pair in a pool of workers using parallel processing.
"""
from multiprocessing import Pool

from gaincalc import calc_weights_onepair
    

def parallel_wrapper(func, maparg, *args, **kwargs):
    def wrapped(maparg):
        return func(maparg, *args, **kwargs)
    return wrapped
    
    
if __name__ == '__main__':
    wrapped_gaingcalc_onepair = parallel_wrapper(
        calc_weights_onepair,
        affectedvar,
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
        
    pool = Pool(processes=4)              # start 4 worker processes
    result = pool.map(wrapped_gaingcalc_onepair,
                      weightcalcdata.affectedvarindexes)
    
