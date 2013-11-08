# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:04:23 2012

@author: St Elmo Wilken
"""

"""Import classes"""
from localGainCalculator import localgains
import numpy as np
from numpy import array, transpose
import networkx as nx


class formatmatrix:
    """This class should format the input connection, gain and variable name matrices.
    This should make it such that you will never have to call localGainCalculator.
    For self made test systems you will have numberofdummyvariables not equal to 0;
    this is especially true if you have recycle streams.
    Additionally, you might want to run formatDiffMatrixForRGA if you have more runs
    than inputs otherwise RGABristol will not work properly.
    
    This class will also use "scale" the nodes of out-vertex == 1 nodes. """
    
    def __init__(self, locationofconnections, locationofstates, numberofruns, numberofdummyvariables, partialcorrelation=False):
        """This class will assume you input a connection matrix (ordered according 
        to the statematrix) with the numberofdummy variables the first N variables
        which are dummy variables to be stripped. """
        
        self.initialiseSystem(locationofconnections, locationofstates, numberofruns,partialcorrelation)
        self.removeDummyVariables(numberofdummyvariables,partialcorrelation) #can be zero!
        self.addforwardScale()
        self.addbackwardScale()
           
    def initialiseSystem(self, locationofconnections, locationofstates, numberofruns,partialcorrelation=False):
        """This method should create the orignal gain matrix (incl dummy gains)
        and the original connection matrix"""
        if partialcorrelation is False:
            original = localgains(locationofconnections, locationofstates, numberofruns)
            self.originalgain = original.linlocalgainmatrix
            self.originaln =original.n #number of rows or columns of gain matrix
            self.variablelist = original.variables
            self.originaldiff = original.localdiffmatrix
            self.originalconnection = original.connectionmatrix
        else:
            original = localgains(locationofconnections, locationofstates, numberofruns,partialcorrelation)
            self.originalgain = original.partialcorrelationmatrix
            self.variablelist = original.variables
            self.originalconnection = original.connectionmatrix
            
    def removeDummyVariables(self, numberofdummyvariables,partialcorrelation=False):
        """This method assumed the first variables up to numberofdummyvariables
        are the dummy variables"""
        
        #fix the list of variables
        if partialcorrelation is False:
            self.nodummyvariablelist = [] #necessary for a list copy
            self.nodummyvariablelist.extend(self.variablelist)
            self.nodummygain = self.originalgain.copy()
            self.nodummyconnection = self.originalconnection.copy()
            self.nodummydiff = self.originaldiff.copy()
            for index in range(numberofdummyvariables):
                self.nodummyvariablelist.pop(0)
                self.nodummygain = np.delete(self.nodummygain,0,0)        
                self.nodummygain = np.delete(self.nodummygain,0,1)
                self.nodummydiff = np.delete(self.nodummydiff,0,0) #you don't want to delete "runs" i.e. columns from the difference matrix       
                self.nodummyconnection = np.delete(self.nodummyconnection,0,0)        
                self.nodummyconnection = np.delete(self.nodummyconnection,0,1)
                
            [r, c] = self.nodummyconnection.shape
            self.nodummyN = r
        else:
            self.nodummyvariablelist = [] #necessary for a list copy
            self.nodummyvariablelist.extend(self.variablelist)
            self.nodummygain = self.originalgain.copy()
            self.nodummyconnection = self.originalconnection.copy()
            for index in range(numberofdummyvariables):
                self.nodummyvariablelist.pop(0)
                self.nodummygain = np.delete(self.nodummygain,0,0)        
                self.nodummygain = np.delete(self.nodummygain,0,1)     
                self.nodummyconnection = np.delete(self.nodummyconnection,0,0)        
                self.nodummyconnection = np.delete(self.nodummyconnection,0,1)
                
            [r, c] = self.nodummyconnection.shape
            self.nodummyN = r            
        
    def formatDiffMatrixForRGA(self, numberofinputs):
        """This method will format the difference matrix such that the RGA method 
        getOpenLoopGain will work properly. It assumes that the no-dummy difference 
        matrix steps each input individually.
        
        This method will create a 2D array where the number of inputs = number of
        columns and the number of rows equals the number of input + output variables.
        Everything should still be ordered. """
        
        self.nodummyRGAformatDiff = []
        
        for ncol in range(numberofinputs):
            self.nodummyRGAformatDiff.append(self.nodummydiff[:,ncol])
            
        self.nodummyRGAformatDiff = transpose(array(self.nodummyRGAformatDiff).reshape(numberofinputs,-1))
        #this works
        
    def addforwardScale(self):
        """This method should add a unit gain node to all nodes with an out-degree
        of 1; now all of these nodes should have an out-degree of 2. Therefore
        all nodes with pointers should have 2 or more edges pointing away from 
        them.
        
        It uses the no dummy variables to construct these gain, connection
        and variable name matrices. """

        M = nx.DiGraph()
        #construct the graph with connections
        for u in range(self.nodummyN):
            for v in range(self.nodummyN):
                if (self.nodummyconnection[u, v] != 0):
                    M.add_edge(self.nodummyvariablelist[v], self.nodummyvariablelist[u], weight = self.nodummygain[u,v])
        
        
        #now add connections where out degree == 1
        counter = 1
        
        for node in M.nodes():
            if M.out_degree(node) == 1:
                nameofscale = 'DV'+str(counter)
                M.add_edge(node, nameofscale, weight = 1.0)
                counter = counter + 1
                
                

        self.scaledforwardconnection = transpose(nx.to_numpy_matrix(M, weight = None))
        self.scaledforwardgain = transpose(nx.to_numpy_matrix(M, weight = 'weight'))
        self.scaledforwardvariablelist = M.nodes() #i sincerely hope this works!... After some testing, I think it does!!!
               
    def addbackwardScale(self):
        """This method should add a unit gain node to all nodes with an out-degree
        of 1; now all of these nodes should have an out-degree of 2. Therefore
        all nodes with pointers should have 2 or more edges pointing away from 
        them.
        
        It uses the no dummy variables to construct these gain, connection
        and variable name matrices. 
        
        Additionally, this method transposes the original no dummy variables to
        generate the reverse option. """

        M = nx.DiGraph()
        transposedconnection = transpose(self.nodummyconnection)
        transposedgain = transpose(self.nodummygain)
        
        #construct the graph with connections
        for u in range(self.nodummyN):
            for v in range(self.nodummyN):
                if (transposedconnection[u, v] != 0):
                    M.add_edge(self.nodummyvariablelist[v], self.nodummyvariablelist[u], weight = transposedgain[u,v])
        
        
        #now add connections where out degree == 1
        counter = 1
        
        for node in M.nodes():
            if M.out_degree(node) == 1:
                nameofscale = 'DV'+str(counter)
                M.add_edge(node, nameofscale, weight = 1.0)
                counter = counter + 1
                
                

        self.scaledbackwardconnection = transpose(nx.to_numpy_matrix(M, weight = None))
        self.scaledbackwardgain = transpose(nx.to_numpy_matrix(M, weight = 'weight'))
        self.scaledbackwardvariablelist = M.nodes() #i sincerely hope this works!... After some testing, I think it does!!!    
