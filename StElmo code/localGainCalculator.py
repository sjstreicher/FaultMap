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
        
        if partialcorrelation is False:
            self.createConnectionMatrix(locationofconnections)
            self.createLocalChangeMatrix(locationofstates)
            self.createLocalDiffmatrix(numberofruns)
            self.createLinearLocalGainMatrix(numberofruns)  
        else:
            self.createConnectionMatrix(locationofconnections)
            self.createCorrelationGainMatrix(locationofstates)
                
    def normaliseGainMatrix(self, inputmatrix):
        """This method normalises the absolute value of the input matrix
        in the columns i.e. all columns will sum to 1"""
            
        [r, c] = inputmatrix.shape
        inputmatrix = abs(inputmatrix) #doesnt affect eigen
        normalisedmatrix = []
        
        for col in range(c):
            colsum = float(sum(inputmatrix[:, col]))
            for row in range(r):
                if (colsum != 0):
                    normalisedmatrix.append(inputmatrix[row, col] / colsum) #this was broken! fixed now...
                else:
                    normalisedmatrix.append(0.0)
                        
        normalisedmatrix = transpose(array(normalisedmatrix).reshape(r, c))
        return normalisedmatrix
                
    def createLocalDiffmatrix(self, numberofruns):
        """This method calculates the deviation of each run from the steady state
        value of that variable. It is assumed that the steady state system is the
        last column of values in the constructor input location of states."""
        
        self.localdiffmatrix = []        
        for row in range(self.n):
            for col in range(numberofruns - 1):
                temp = self.localchangematrix[row, col] - self.localchangematrix[row, numberofruns - 1]
                self.localdiffmatrix.append(temp)
        
        self.localdiffmatrix = array(self.localdiffmatrix).reshape(self.n, -1)
        
    def createLinearLocalGainMatrix(self, numberofruns):
        """This method creates a local gain matrix using the following method:
           output_change|exp1 = gain1*input_change1|exp1 + gain2*input_change2|exp1 + etc...
           output_change|exp2 = gain1*input_change1|exp2 + gain2*input_change2|exp2 + etc...
           A least squares routine is used to determine the gains which result in the smallest error.
           The connectivity is determined by the connection matrix"""
        
        self.linlocalgainmatrix = array(zeros((self.n, self.n)))  #initialise the linear local gain matrix
        for row in range(self.n):
            index = self.connectionmatrix[row, :].reshape(1, self.n)
            if (max(max(index)) > 0): #crude but it works...    
                compoundvec = self.localdiffmatrix[row, :].reshape(numberofruns - 1, 1)
                #now you need to get uvec so that you may calculate the aprox gains
                #note: rows == number of experiments       
                for position in range(self.n):
                    if index[0, position] == 1:
                        temp = self.localdiffmatrix[position, :].reshape(-1, 1) # dummy variable
                        compoundvec = hstack((compoundvec, temp))
                    else:
                        pass #do nothing as the index will sort out the order of gain association
                yvec = compoundvec[:, 0].reshape(-1, 1)
                uvec = compoundvec[:, 1:]       
                localgains = np.linalg.lstsq(uvec, yvec)[0].reshape(1, -1)
                tempindex = 0        
                for position in range(self.n):
                    if index[0, position] == 1:
                        self.linlocalgainmatrix[row, position] = localgains[0, tempindex]
                        tempindex = tempindex + 1
                    else:
                        pass #do nothing as the index will sort out the order of gain association
            else:
                pass #everything works
        
    def createLocalChangeMatrix(self, locationofstates):
        """This method imports the states of the variables during the different 
        test runs (it is assumed steady state is the final column) octave inserts 
        an empty space in front of every row so this program will assume this pattern for 
        all inputs this program will assume the base case is the last column of data"""
        

        fromfile = csv.reader(open(locationofstates), delimiter=' ')
        self.localchangematrix = []
        for line in fromfile:
            linefixed = line[1:] #to get rid of a white space preceeding every line
            for element in linefixed:
                self.localchangematrix.append(float(element))
        self.localchangematrix = array(self.localchangematrix).reshape(len(self.variables), -1)
            
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
        self.variables = fromfile.next()[1:] #gets rid of that first space. Now the variables are all stored
        self.connectionmatrix = []
        for row in fromfile:
            col = row[1:] #this gets rid of the variable name on each row (its there to help create the matrix before its read in)
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
        
        fromfile = csv.reader(open(statesloc), delimiter = ' ')
        dataholder = []

        for row in fromfile:
            rowfix = row[1:]
            for element in rowfix:
                dataholder.append(float(element))
        
        self.inputdata = array(dataholder).reshape(-1, self.n)
        
        self.correlationmatrix = array(np.empty((self.n, self.n)))
        
        for i in range(self.n):
            for j in range(self.n):    
                temp = pearsonr(self.inputdata[:, i], self.inputdata[:, j])[0]
                if temp is 'nan':
                    self.correlationmatrix[i,j] = temp
                else: #this is just to guarantee that the matrix is not populated by nan elements
                    self.correlationmatrix[i,j] = random()*0.01 #this is here just in case: adds a small random number
                    #self.correlationmatrix[i,j] = temp
        
        P = np.linalg.inv(self.correlationmatrix)
        self.partialcorrelationmatrix = array(np.empty((self.n, self.n)))
        
        for i in range(self.n):
            for j in range(self.n):
                if self.connectionmatrix[i,j] == 1:
                    temp = -1*P[i,j]/( abs( (P[i,i]*P[j,j]) )**0.5 )
                    self.partialcorrelationmatrix[i,j] = temp
                else:
                    self.partialcorrelationmatrix[i,j] = 0
                 
                    
                
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
