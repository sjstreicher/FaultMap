'''
Created on 05 May 2012

@author: St Elmo Wilken
'''
"""Import classes"""
#from visualise import visualiseOpenLoopSystem
from gainRank import gRanking
from numpy import array, transpose, arange, empty
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import permutations, izip
import csv


class loopranking:
    """This class will:
    1) Rank the importance of nodes in a system with control
    1.a) Use the local gains to determine importance
    1.b) Use partial correlation information to determine importance
    2) Determine the change of importance when variables change
    """
    
    def __init__(self, fgainmatrixC, fvariablenamesC, fconnectionmatrixC, bgainmatrixC, bvariablenamesC, bconnectionmatrixC, nodummyvariablelistC, fgainmatrix, fvariablenames, fconnectionmatrix, bgainmatrix, bvariablenames, bconnectionmatrix, alpha = 0.35 ):
        """This constructor will:
        1) create a graph with associated node importances based on local gain information
        2) create a graph with associated node importances based on partial correlation data"""
        
        self.forwardgain = gRanking(self.normaliseMatrix(fgainmatrixC), fvariablenamesC)      
        self.backwardgain = gRanking(self.normaliseMatrix(bgainmatrixC), bvariablenamesC)
        self.createBlendedRanking(nodummyvariablelistC, alpha)
        
    def createBlendedRanking(self, nodummyvariablelist, alpha = 0.35):
        """This method will create a blended ranking profile of the object"""
        
        self.variablelist = nodummyvariablelist
        
        self.blendedranking = dict()
        for variable in nodummyvariablelist:
            self.blendedranking[variable] = (1 - alpha) * self.forwardgain.rankDict[variable] + (alpha) * self.backwardgain.rankDict[variable]
            
        slist = sorted(self.blendedranking.iteritems(), key = itemgetter(1), reverse=True)
        for x in slist:
            print(x)
        print("Done with Controlled Importances")
        
    def normaliseMatrix(self, inputmatrix):
        """This method normalises the absolute value of the input matrix
        in the columns i.e. all columns will sum to 1
        
        It also appears in localGainCalculator but not for long! Unless I forget
        about it..."""
        
        [r, c] = inputmatrix.shape
        inputmatrix = abs(inputmatrix) #Does not affect eigenvalues
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
 
#    def displayControlImportances(self,nocontrolconnectionmatrix, controlconnectionmatrix ):
#        """This method will create a graph containing the 
#        connectivity and importance of the system being displayed.
#        Edge Attribute: color for control connection
#        Node Attribute: node importance        
#        """
#        
#        ncG = nx.DiGraph()
#        n = len(self.variablelist)
#        for u in range(n):
#            for v in range(n):
#                if nocontrolconnectionmatrix[u,v] == 1:
#                    ncG.add_edge(self.variablelist[v], self.variablelist[u])
#        
#        edgelistNC = ncG.edges()
#        
#        self.controlG = nx.DiGraph()
#        
#        for u in range(n):
#            for v in range(n):
#                if controlconnectionmatrix[u,v] == 1:
#                    if (self.variablelist[v], self.variablelist[u]) in edgelistNC:
#                        self.controlG.add_edge(self.variablelist[v], self.variablelist[u], controlloop = 0)
#                    else:
#                        self.controlG.add_edge(self.variablelist[v], self.variablelist[u], controlloop = 1)
#        
#        
#        for node in self.controlG.nodes():
#            self.controlG.add_node(node, nocontrolimportance = self.blendedrankingNC[node] , controlimportance = self.blendedranking[node])
#        
#        plt.figure("The Controlled System")
#        nx.draw_circular(self.controlG)
#        
#    def showAll(self):
#        """This method will show all figures"""
#        
#        plt.show()
#        
#    def exportToGML(self):
#        """This method will just export the control graphs
#        to a gml file"""
#        
#        try:
#            if self.controlG:
#                print("controlG exists")
#                nx.write_gml(self.controlG, "controlG.gml")
#        except:
#            print("controlG does not exist")
        
                     
        
























