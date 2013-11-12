# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:33:40 2012

@author: St Elmo Wilken
"""

"""Import classes and modules"""
from numpy import ones, argmax
from numpy import linalg as linCalc
 
class gRanking:
    """This class should calculate the importance of variables based on their
    first order local gains."""
    
    def __init__(self,localgainmatrix, listofvariables):
        """The constructor creates a rankings dictionary with the variables as
        keys and the normlised rank as value."""
        
        self.gMatrix = localgainmatrix
        self.gVariables = listofvariables 
        self.constructRankArray()  
    
    def constructRankArray(self):
        """This method constructs the ranking dictionary using the eigenvector
        approach i.e. Ax = x where A is the local gain matrix"""
        
        self.n = len(self.gMatrix) #length of gain matrix = number of nodes
        S = (1.0/self.n)*ones((self.n,self.n))
        m = 0.15
        self.M = (1-m)*self.gMatrix + m*S #basic page rank algorithm
        [eigVal, eigVec] = linCalc.eig(self.M) #calc eigenvalues, eigenvectors as usual
        
        maxeigindex = argmax(eigVal)
        self.maxeig = eigVal[maxeigindex].real # store value for downstream checking

        self.rankArray = eigVec[:,maxeigindex] #cuts array into the eigenvector corrosponding to the eigenvalue above
        self.rankArray = (1/sum(self.rankArray))*self.rankArray #this is the 1 dimensional array composed of rankings (normalised)
        self.rankArray = self.rankArray.real #to take away the useless +0j part...
                
        #create a dictionary of the rankings with their respective nodes ie {NODE:RANKING}
        self.rankDict = dict(zip(self.gVariables,self.rankArray))
        #print(self.rankDict) this works. now need to rearrange the rank sizes to corrospond to the drawing...
        

                


    
       
        
        
