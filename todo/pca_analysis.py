# -*- coding: utf-8 -*-
"""
Performs basic PCA analysis

Input data format is CSV file with labels in header row
timstamps in first column (which can be used as an index - preferably UNIX time)
and data after that

Created on Wed Jun 15 16:48:24 2016

@author: Simon Streicher
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.decomposition import PCA

datafilename = "C:/Repos/testdata/pca_analysis/data_raw_realtime_labels.csv"
sets_filenames = {'datafile': datafilename}

datasets = ['datafile']
process_datasets = {'datafile': True}
timelabel_datasets = {'datafile': 'Time'}

resample = False
# Resampling period in seconds
resampling_period = 60
resample_string = str(resampling_period) + 'S'

truncate = False
start_date = 1424131200
end_date = 1424131200

# Initiate empty dataframes
dfs = dict.fromkeys(datasets)

for dataset in datasets:
    if process_datasets[dataset]:
        datadf = pd.read_csv(sets_filenames[dataset])

#        datadf = datafunctions.convert_dates(
#            datadf, timelabel_datasets[dataset])
        datadf = datadf.set_index([timelabel_datasets[dataset]])

        # Force numeric columns to numeric
        for key in datadf.keys():
            datadf[key] = pd.to_numeric(
                datadf[key], errors='coerce')

        if resample:
            datadf = \
                datadf.resample(resample_string).mean()
        if truncate:
            datadf = datadf.loc[start_date:end_date]

        dfs[dataset] = datadf

        with open('{}.pickle'.format(dataset), 'wb') as f:
            pickle.dump(datadf, f)
    else:
        # Load dataframes back from pickled objects
        with open('{}.pickle'.format(dataset), 'rb') as f:
            dfs[dataset] = pickle.load(f)

testset = dfs[datasets[0]]

# TODO: Standardize data option

testarray = np.array(testset)
pca = PCA(n_components=0.95)
pca.fit(testarray)
components = pca.fit_transform(testarray)
print pca.explained_variance_ratio_



