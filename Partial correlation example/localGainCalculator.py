# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 23:12:10 2012

@author: St Elmo Wilken
"""

"""Import classes and module"""
from numpy import array, transpose, zeros, hstack
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from random import random
from math import isnan

class localgains:
    """This class:
        1) Imports the connection matrix from file
        2) Imports the state matrix from file
        3) Constructs the linear local gain matrix"""
    
    def __init__(self, locationofconnections, locationofstates, numberofruns, partialcorrelation=False):
        """This constructor creates matrices in memory of the connection, 
        local gain and local diff (from steady state) inputs"""
        self.createConnectionMatrix(locationofconnections)
        self.createCorrelationGainMatrix(locationofstates)
            
    def createConnectionMatrix(self, nameofconn):
        """This method imports the connection scheme for the data. 
        The format should be: 
        empty space, var1, var2, etc... (first row)
        var1, value, value, value, etc... (second row)
        var2, value, value, value, etc... (third row)
        etc...
        
        Value is 1 if column variable points to row variable (causal relationship)
        Value is 0 otherwise
        
        This method also stores the names of all the variables in the connection matrix.
        It is important that the order of the variables in the connection matrix match
        those in the data matrix"""
        
        fromfile = csv.reader(open(nameofconn))
        self.variables = fromfile.next()[1:]  # Gets rid of that first space. Now the variables are all stored
        self.connectionmatrix = []
        for row in fromfile:
            col = row[1:] # This gets rid of the variable name on each row (its there to help create the matrix before its read in)
            for element in col:
                if element == '1':
                    self.connectionmatrix.append(1)
                else:
                    self.connectionmatrix.append(0)
        
        self.n = len(self.variables)
        self.connectionmatrix = array(self.connectionmatrix).reshape(self.n, self.n)

    def createCorrelationGainMatrix(self, statesloc):
        """This method strives to calculate the local gains in terms of the correlation
        between the variables. It uses the partial correlation method (Pearson's 
        correlation)."""
        
        fromfile = csv.reader(open(statesloc), delimiter = ',')
        dataholder = []

        for row in fromfile:
            rowfix = row
            for element in rowfix:
                dataholder.append(float(element))
        
        print self.n
        print np.shape(dataholder)
        
        self.inputdata = np.reshape(dataholder, (-1, self.n))
        
        self.correlationmatrix = array(np.empty((self.n, self.n)))
        
        for i in range(self.n):
            for j in range(self.n):    
                temp = pearsonr(self.inputdata[:, i], self.inputdata[:, j])[0]
                if isnan(temp):
                    self.correlationmatrix[i,j] = random()*0.01
                else:
                    self.correlationmatrix[i,j] = temp
        
        P = np.linalg.inv(self.correlationmatrix)
        self.partialcorrelationmatrix = array(np.empty((self.n, self.n)))
        
        for i in range(self.n):
            for j in range(self.n):
                if self.connectionmatrix[i,j] == 1:
                    temp = -1*P[i,j]/( abs( (P[i,i]*P[j,j]) )**0.5 )
                    self.partialcorrelationmatrix[i,j] = temp
                else:
                    self.partialcorrelationmatrix[i,j] = 0
                 
                    
                
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
