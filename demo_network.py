# -*- coding: utf-8 -*-
"""
Created on Thu Mar 06 14:18:54 2014

@author: Simon
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
logging.basicConfig(level=logging.INFO)

import os

filesloc = json.load(open('config.json'))
saveloc = os.path.expanduser(filesloc['saveloc'])

testgraph = nx.DiGraph()

variables = ['a', 'b', 'c']

connections = np.matrix([[0, 1, 1],
                         [1, 0, 0],
                         [0, 1, 0]])

gainmatrix = np.matrix([[0.55, 0.25, 0.36],
                        [0.14, 0.32, 0.47],
                        [0.26, 0.89, 0.44]])

for col, colvar in enumerate(variables):
    for row, rowvar in enumerate(variables):
        if (connections[row, col] != 0):
            # The node order is source, sink according to
            # the convention that columns are sources and rows are sinks
            testgraph.add_edge(colvar, rowvar, weight=gainmatrix[row, col])

nx.write_gml(testgraph, os.path.join(saveloc, "testgraph.gml"))
nx.draw(testgraph)
plt.show()