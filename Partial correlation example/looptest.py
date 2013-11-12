'''
Created on 05 May 2012

@author: St Elmo Wilken
'''
"""This class will be used to run controlranking"""

"""Import classes"""
from controlranking import loopranking
from formatmatrices import formatmatrix
from numpy import array, transpose, arange, empty
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter


datamatrix = formatmatrix("connectionsTEcontrol.csv","data.csv",0,0,partialcorrelation=True)
controlmatrix = loopranking(datamatrix.scaledforwardgain, datamatrix.scaledforwardvariablelist, datamatrix.scaledforwardconnection, datamatrix.scaledbackwardgain, datamatrix.scaledbackwardvariablelist, datamatrix.scaledbackwardconnection, datamatrix.nodummyvariablelist, [], [], [], [], [], [])

#controlmatrix.displayControlImportances([], datamatrix.nodummyconnection)
#
#controlmatrix.showAll()
#controlmatrix.exportToGML()
        
    
