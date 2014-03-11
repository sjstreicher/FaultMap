# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:29:33 2014

@author: Simon
"""

import numpy as np
import tables as tb
import json

# Define the datafile name as defined in the config file
# This is also used to name the created file name and filename
datafile = 'data_pressetstep_mod'
# Define any connection file with the same variables as the data file columns
# Only needed when creating columns
#connectionfile = 'all_connections_measonly'

filesloc = json.load(open('config.json'))
# Load the csv file
tags_tsdata = filesloc[datafile + "_csv"]
# Defines the destination of the created hdf5 file
saveloc = filesloc['savelocation']
#connection_loc = filesloc[connectionfile]

# Create a description instance
# Only needed when creating a table
#keys, _ = create_connectionmatrix(connection_loc)
#for index, name in enumerate(keys):
#    keys[index] = name.replace(" ", "_")
#value = ['float64'] * len(keys)
#data_description = np.dtype(zip(keys, value))

hdf5writer = tb.open_file(saveloc + datafile + '.h5', 'w')
data = np.genfromtxt(tags_tsdata, delimiter=',')

#table = hdf5writer.create_table(hdf5writer.root, description=data_description,
#                                name=datafile,
#                                title=datafile)
                                
array = hdf5writer.create_array(hdf5writer.root, datafile, data)
            
#table.append(data)
#table.flush()
array.flush()
hdf5writer.close()