"""
Created on Mon Feb 24 15:27:21 2014

@author: Simon Streicher
"""

from ranking.formatmatrices import buildgraph
import networkx as nx
import numpy as np
import json
import os

filesloc = json.load(open('config.json'))
saveloc = os.path.expanduser(filesloc['saveloc'])

def fullconn_equal():
    """Creates a fully connected 5x5 digraph with unit weights
    on all edges.
    
    """
 
    variables = ['X 1', 'X 2', 'X 3', 'X 4', 'X 5']
    
    connections =  np.ones((5, 5))
    gainmatrix = np.ones((5, 5))
    
    testgraph = buildgraph(variables, gainmatrix, connections)

    nx.write_gml(testgraph, os.path.join(saveloc, 'testgraphs',
                                         "fullconn_equal.gml"))
    nx.draw(testgraph)
    
    return connections, gainmatrix, variables, testgraph