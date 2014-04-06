# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:29:33 2014

@author: Simon
"""

import numpy as np
import tables as tb
import json
from config_setup import runsetup
import os

mode = 'plants'
case = 'epu5_compressor'

saveloc, casedir, _ = runsetup(mode, case)
caseconfig = json.load(open(os.path.join(casedir, case + '.json')))
scenarios = caseconfig['scenarios']

scenario = scenarios[0]

tags_tsdata = os.path.join(casedir, 'data',
                           caseconfig[scenario]['data'])
dataset = caseconfig[scenario]['dataset']

# Define the datafile name as defined in the config file
# This is also used to name the created file name and filename
#datafile = 'data_raw'
# Define any connection file with the same variables as the data file columns
# Only needed when creating columns
#connectionfile = 'all_connections_measonly'


#connection_loc = filesloc[connectionfile]

# Create a description instance
# Only needed when creating a table
#keys, _ = create_connectionmatrix(connection_loc)
#for index, name in enumerate(keys):
#    keys[index] = name.replace(" ", "_")
#value = ['float64'] * len(keys)
#data_description = np.dtype(zip(keys, value))

hdf5writer = tb.open_file(saveloc + dataset + '.h5', 'w')
data = np.genfromtxt(tags_tsdata, delimiter=',')

#table = hdf5writer.create_table(hdf5writer.root, description=data_description,
#                                name=datafile,
#                                title=datafile)
                                
array = hdf5writer.create_array(hdf5writer.root, dataset, data)
            
#table.append(data)
#table.flush()
array.flush()
hdf5writer.close()