# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 17:09:34 2012

@author: St Elmo Wilken
"""
from gainRank import gRanking
class nRanking(gRanking):
    
    def __init__(self,inMat1,inMat2,alpha,matIV):
        from numpy import array        
        self.gMatrix = array(inMat1) #feed in a normalised gain matrix n*n array
        self.gVariables = inMat2 #feed in ordered variables wrt gMatrix 1*n array
        self.alpha = alpha #feed in the weighting of the intrinsic value analysis relative to the gain analysis
        self.intrinsicValue = matIV # feed in the intrinsic value of the variables 1*n array
        #methods
        self.constructAdjacencyArray()   
        self.constructIntrinsicValueArray()
        self.constructRankArrayWithIntrinsicValues()
        
    def constructAdjacencyArray(self):
        self.n = len(self.gMatrix) #number of rows of gain matrix = number of nodes
        from numpy import array, zeros
        self.adjacencyArray = array(zeros((self.n,self.n)))
        for i in range(self.n):
            for j in range(self.n):
                if self.gMatrix[i,j] != 0:
                    self.adjacencyArray[i,j] = 1
    
    def constructIntrinsicValueArray(self): #also normalise it
        #the matIV feed tells you which variables are more important
        #basically a list of factors
        from numpy import array, zeros 
        self.intrinsicValueArray = array(zeros((self.n,self.n))) 
        for i in range(self.n):
            for j in range(self.n):
                self.intrinsicValueArray[i,j] = self.adjacencyArray[i,j]*self.intrinsicValue[i]
        #need to normalise the result of above
        self.totals = [sum(self.intrinsicValueArray[:,j]) for j in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                self.intrinsicValueArray[i,j] = self.intrinsicValueArray[i,j]*(1.0/self.totals[j])   
    
    def constructRankArrayWithIntrinsicValues(self):
        from numpy import ones, argmax
        from numpy import linalg as linCalc
        # M = (1-m)*A + m*S
        # A = (1-alpha)*B + alpha*C
        # B = normalised gain matrix
        # C = normalised intrinsic value matrix
        # S = gets rid of sub stochasticity for rows of all 0
        self.A = (1-self.alpha)*self.gMatrix + self.alpha*self.intrinsicValueArray
        self.S = (1.0/self.n)*ones((self.n,self.n))
        m = 0.15
        self.M = (1-m)*self.A+ m*self.S #basic page rank algorithm
        [eigVal, eigVec] = linCalc.eig(self.M) #calc eigenvalues, eigenvectors as usual
        
        maxeigindex = argmax(eigVal)
        self.maxeig = eigVal[maxeigindex].real # store value for downstream checking

        self.rankArray = eigVec[:,maxeigindex] #cuts array into the eigenvector corrosponding to the eigenvalue above
        self.rankArray = (1/sum(self.rankArray))*self.rankArray #this is the 1 dimensional array composed of rankings (normalised)
        self.rankArray = self.rankArray.real #to take away the useless +0j part...


if __name__ == "__main__":
        
    mat1 = [[0,0,0,0,0,0,0,0,0,0,0,0.5,0,0],[0,0,0,0,0,0,0,0,0,0,0.5,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0.5,0,0],[0,0,0,0,0,0,0,0,0,0,0.5,0,0,0],[1.0/3,0,1.0/3,0,0,0,1.0/3,0,0,0,0,0,0,1.0/3],[1.0/3,0,1.0/3,0,1,0,1.0/3,0,0,0,0,0,0,1.0/3],[0,1,0,1,0,0.5,0,0,0,0,0,0,1,0],[1.0/3,0,1.0/3,0,0,0.5,0,0,0,0,0,0,0,1.0/3],[0,0,0,0,0,0,1.0/3,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0.5,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0.5,0,0,0,0],[0,0,0,0,0,0,0,0,0.5,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0.5,0,0,0,0]]   
    mat2 = ["T1","F1","T2","F2","R1","X1","F3","T3","F4","T4","F6","T6","F5","T5"]
    mat3 = [4,1,4,1,1,1,1,4,1,4,1,4,1,4] #all the temperature variable are 4 times more important than the other ones (why? for safety reasons)
    test = nRanking(mat1,mat2,0.5,mat3)
    print(test.rankArray)

    test.showConnectRank()
    
    
    
    
