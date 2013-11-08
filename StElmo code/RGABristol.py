# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 00:45:09 2012

@author: St Elmo Wilken
"""

"""Import classes and modules"""
from numpy import array, transpose, argmax, copy, delete
import numpy as np

class RGA:
    """This class implements the RGA method"""
    
    def __init__(self, variables, localdiffs, numberofinputs, positioncontrol):
        """This constructor creates the recommended input-output pairings
        of the RGA method. There are optional display methods. """
        
        self.getOpenLoopGainArray(localdiffs, numberofinputs, variables, positioncontrol)
        self.calculateBristolmatrix()
        self.calculateBristolpairingsMax(self.vars, numberofinputs)
        self.calculateBristolpairingsHalf(self.vars, numberofinputs)
    
    def getOpenLoopGainArray(self, localdiffs, numberofinputs, variables ,controlposition = None):
        """This method calculates the open loop gain matrix as used by the 
        RGA method. This assumes that columns are inputs and rows are outputs.
        
        This method assumes that each input is changed individually i.e.
        you will only have input rows with only one non-zero value.
        
        A difference matrix is read in. This should be formatted such that its 
        dimensions are: number of rows = number of variables (i.e. number of
        inputs + number of outputs) and number of columns = number of inputs 
        (because you assume that you change each input only once.
        
        The result of this method is an open loop gain matrix with dimensions:
        number of rows = number of outputs
        number of columns = number of inputs
        
        Finally, this method takes as a parameter positioncontrol (which is a list.) 
        This parameter tells the method which variables are to be controlled i.e.
        it cuts the other variables out of the open loop gain matrix. The parameter
        has a default setting equal to None i.e. no rows are deleted. """
               
        [r, c] = localdiffs.shape
        outputs = localdiffs[numberofinputs:r, :]
        inputs = localdiffs[:numberofinputs, :]#array of inputs
        self.openloopmatrix = []
        for colin,colout in zip(transpose(inputs),transpose(outputs)): #such that you iterate over every column
            change = max(colin)
            colofgain = colout/change
            self.openloopmatrix.append(colofgain)
        self.openloopmatrix = array(self.openloopmatrix).reshape(numberofinputs,-1)
        self.openloopmatrix = transpose(self.openloopmatrix)
        #now split the resulting matrix into the portions you will actually use
        [r, c] = self.openloopmatrix.shape
        tempmatrix = []

        if controlposition != None:
            self.vars = variables[:numberofinputs] #all the inputs
            for row in range(r):
                if variables[row+numberofinputs] in controlposition:
                    tempmatrix.append(self.openloopmatrix[row, :])
                    self.vars.append(variables[row+numberofinputs])
            self.openloopmatrix = array(tempmatrix).reshape(-1, numberofinputs)
        else:
            self.vars = variables
        self.openloopmatrix = transpose(self.openloopmatrix)
        #up to here works perfectly. the last transpose fixes a slight definition error made earlier   
                    
    def calculateBristolmatrix(self):
        """This method actually calculates the relative gain array.
        It uses the standard method for square open loop gain matrices
        and for non square open loop gain matrices the pseudo-inverse
        is calculated (and then the standard method is used)."""
        
        [r,c] = self.openloopmatrix.shape #because not necessarily square
        self.bristolmatrix = []

        if (r==c): #square
            if (np.linalg.det(self.openloopmatrix) != 0):
                R = np.transpose(np.linalg.inv(self.openloopmatrix))
                for gij, rij in zip(self.openloopmatrix.flat, R.flat):
                    self.bristolmatrix.append(gij*rij)
                self.bristolmatrix = array(self.bristolmatrix).reshape(r,r)
            else:
                print("Sorry, the open loop gain matrix is singular")
        else: #seems pinv calculates the moore-penrose inverse... so left or right inverse is not a problem
            R = np.linalg.pinv(self.openloopmatrix).transpose()
            for gij, rij in zip(self.openloopmatrix.flat, R.flat):
                self.bristolmatrix.append(gij*rij)
            self.bristolmatrix = array(self.bristolmatrix).reshape(r,c)
        
    def calculateBristolpairingsMax(self,vararrs,numofinputs):
        """This method determines which variables should be paired using the RGA.
        It assumed that each input WILL be paired and as such, it looks for the
        biggest value in each column and pairs accordingly.
        This will allow only a single output (the best controllable case) to
        map to a single output. """
        
        self.inputvars = vararrs[:numofinputs]        
        self.outputvars = vararrs[numofinputs:]
        tempoutputs = []
        tempoutputs.extend(self.outputvars)
        tempbristol = self.bristolmatrix.copy()
        pairedvariables = []
        [r, c] = tempbristol.shape
        count = 0
        for row in range(r):
            pos = argmax(tempbristol[row, :])
            pairedvariables.append(tempoutputs[pos])
            pairedvariables.append(self.inputvars[count])
            tempbristol = delete(tempbristol, pos, 1)
            tempoutputs.pop(pos)
            count += 1   
        self.pairedvariablesMax = array(pairedvariables).reshape(-1,2)
        
    def calculateBristolpairingsHalf(self,vararrs,numofinputs):
        """This method determines the best pairings of the RGA. It 
        pairs only variables which are sufficiently decoupled ( >= 0.5 in RGA)
        This will allow multiple outputs to be controlled by a single input."""
        
        self.inputvars = vararrs[:numofinputs]        
        self.outputvars = vararrs[numofinputs:]
        tempoutputs = []
        tempoutputs.extend(self.outputvars)
        tempbristol = self.bristolmatrix.copy()
        pairedvariables = []
        [r, c] = tempbristol.shape
        count = 0
        for row in range(r):
            pos = argmax(tempbristol[row, :])
            if tempbristol[row, :][pos] >= 0.5:
                pairedvariables.append(tempoutputs[pos])
                pairedvariables.append(self.inputvars[count])
                tempbristol = delete(tempbristol, pos, 1)
                tempoutputs.pop(pos)
            count += 1
        self.pairedvariablesHalf = array(pairedvariables).reshape(-1,2)
           
        
            
        