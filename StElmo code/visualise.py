# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:52:25 2012

@author: St Elmo Wilken
"""

"""Import classes"""
from numpy import array, transpose, arange, empty
import networkx as nx
import matplotlib.pyplot as plt
from RGABristol import RGA
from gainRank import gRanking
from operator import itemgetter
from itertools import permutations, izip

class visualiseOpenLoopSystem:
    """The class name is not strictly speaking accurate as some calculations are 
    also done herein.
    This class will:
    
        1) Visualise the connectivity and local gain information
        2) Visualise the results of the RGA method
        3) Visualise the results of the eigen-vector approach method
        4) Calculate the edge-weights based on the algorithm
        5) Calculate and display the best pairing scheme according to the
            eigen-method
            
    This class will create several graphs:
    
    G = the directed connection graph with edge attribute 'localgain'
    G1 = RGA recommended pairings using 0.5 criteria. edgecolour attribute
    G2 = RGA recommended pairings using max criteria. edgecolour attribute.
    GGF = Google Gain Forward (scaled) graph with node importance as node attribute 'importance' 
    LGF = Local Gain Forward (scaled) graph with node importance as node attribute 'importance'
    GGB = Google Gain Backward (scaled) graph with node importance as node attribute 'importance'
    LGB = Local Gain Backward (scaled) graph with node importance as node attribute 'importance' 
    NFG = Normal Forward Gain (not scaled) graph with node importance as node attribute 'importance'
    NBG = Normal Backward Gain (not scaled) graph with node importance as node attribute 'importance'
    EBG = Eigen Blended Graph = Eigen approach using LGF and LGB to calculate node attribute 'importance' 
    EBGG = Eigen Blended Google Graph = Eigen approach using GGF and GGB to calculate node attribute 'importance'  
    P = A graph showing the various node importances and the associated edge weights.
        Node Attributes: importanceNormal = blended importance; importanceGoogle = blended importance Google
        Edge Attribute = weight = edge weight according to algorithm 
    F = A graph showing the recommended control pairings with edge attribute 'edgecolour' to indicate a 
        control pair. """
    
    def __init__(self, variables, localdiff, numberofinputs, fgainmatrix, fconnectionmatrix, fvariablenames, bgainmatrix, bconnectionmatrix, bvariablenames, normalgains, normalconnections, controlvarsforRGA=None):
        """This constructor will create an RGABristol object so that you simply
        have to call the display method to see which pairings should be made.
        
        It will also create 6 different ranking systems. Note that variablenames
        is not the same as variables!! There is a formatting difference. 
        
        ASSUME: the first rows are the inputs up to numberofinputs"""
        
        self.bristol = RGA(variables, localdiff, numberofinputs, controlvarsforRGA)
        
        self.forwardgain = gRanking(self.normaliseMatrix(fgainmatrix), fvariablenames)
        self.gfgain = gRanking(self.normaliseMatrix(fconnectionmatrix), fvariablenames)        
        
        self.backwardgain = gRanking(self.normaliseMatrix(bgainmatrix), bvariablenames)
        self.gbgain = gRanking(self.normaliseMatrix(bconnectionmatrix), bvariablenames)
        
        self.normalforwardgain = gRanking(self.normaliseMatrix(normalgains), variables)
        self.normalbackwardgain = gRanking(self.normaliseMatrix(transpose(normalgains)), variables)
        self.normalforwardgoogle = gRanking(self.normaliseMatrix(normalconnections), variables)
        
        self.listofinputs = variables[:numberofinputs]
        self.listofoutputs = variables[numberofinputs:]
        
    def displayConnectivityAndLocalGains(self, connectionmatrix, localgainmatrix, variablenames, nodepositiondictionary=None):
        """This method should display a graph indicating the connectivity of a
        system as well as the local gains calculated by this class. The default
        layout is circular.
        
        It specifically requires an input connection and local gain matrix
        so that you made format them before display. Be careful to make sure
        the variables are ordered correctly i.e. don't do this manually for large
        systems.
        
        It has an optional argument to specify the position of the nodes.
        This should be entered as a dictionary in the format:
        key = node : value = array([x,y])
        
        It will create a graph with an edge attribute called localgain."""
        
        [n, n] = localgainmatrix.shape        
        self.G = nx.DiGraph() #this is convenient
        localgaindict = dict()
        localgaindictformat = dict()
        for u in range(n):
            for v in range(n):
                if (connectionmatrix[u, v] == 1):
                    self.G.add_edge(variablenames[v], variablenames[u], localgain=round(localgainmatrix[u, v]))
                    localgaindict[(variablenames[v], variablenames[u])] = localgainmatrix[u, v]
                    localgaindictformat[(variablenames[v], variablenames[u])] = round(localgainmatrix[u, v], 3)
                    
        posdict = nodepositiondictionary 
        
        if posdict == None:
            posdict = nx.circular_layout(self.G)
    
        plt.figure("Web of connectivity and local gains")
        nx.draw_networkx(self.G, pos=posdict)
        nx.draw_networkx_edge_labels(self.G, pos=posdict, edge_labels=localgaindictformat, label_pos=0.7)
        nx.draw_networkx_edges(self.G, pos=posdict, width=2.5, edge_color='k', style='solid', alpha=0.15)
        nx.draw_networkx_nodes(self.G, pos=posdict, node_color='y', node_size=450)
        plt.axis("off") 
        
    def displayRGA(self, pairingoption=1, nodepositions=None):
        """This method will display the RGA pairings.
        
        It has 2 options of pairings:
            1) pairingoption = 1 (the default) This displays the standard RGA
            pairings where the decision to pair is positive if the relative gain
            array has an element value of more than or equal to 0.5. 
            2) pairingoption = 2 This displays the RGA pairs where each input is
            forced to have a paired output. This is selected by using the maximum 
            value in each column as a pair.
            
        It has an optional parameter to set node positions. If left out
        the default node positions will be circular. """

        if pairingoption == 1:
            pairingpattern = self.bristol.pairedvariablesHalf
            message = "Standard RGA Pairings"
            self.G1 = nx.DiGraph()
            self.G1 = self.G.copy()
            print(pairingpattern)
            self.G1.add_edges_from(self.G1.edges(), edgecolour='k')
            self.G1.add_edges_from(pairingpattern, edgecolour='r')
            #correct up to here
            pairingtuplelist = [(row[0], row[1]) for row in pairingpattern] #what a mission to find this error
            edgecolorlist = ["r" if edge in pairingtuplelist else "k" for edge in self.G1.edges()]
        
                
            if nodepositions == None:
                nodepositions = nx.circular_layout(self.G1)
            
            plt.figure(message)            
            nx.draw_networkx(self.G1, pos=nodepositions)
            nx.draw_networkx_edges(self.G1, pos=nodepositions, width=2.5, edge_color=edgecolorlist, style='solid', alpha=0.5)
            nx.draw_networkx_nodes(self.G1, pos=nodepositions, node_color='y', node_size=450)
            plt.axis('off')
        else:
            pairingpattern = self.bristol.pairedvariablesMax
            message = "Maximum RGA Pairings"
            self.G2 = nx.DiGraph()
            self.G2 = self.G.copy()
            print(pairingpattern)
            self.G2.add_edges_from(self.G2.edges(), edgecolour='k')
            self.G2.add_edges_from(pairingpattern, edgecolour='r')
        #correct up to here
            pairingtuplelist = [(row[0], row[1]) for row in pairingpattern] #what a mission to find this error
            edgecolorlist = ["r" if edge in pairingtuplelist else "k" for edge in self.G2.edges()]
        
                
            if nodepositions == None:
                nodepositions = nx.circular_layout(self.G2)
            
            plt.figure(message)            
            nx.draw_networkx(self.G2, pos=nodepositions)
            nx.draw_networkx_edges(self.G2, pos=nodepositions, width=2.5, edge_color=edgecolorlist, style='solid', alpha=0.5)
            nx.draw_networkx_nodes(self.G2, pos=nodepositions, node_color='y', node_size=450)
            plt.axis('off')
                
    def displayRGAmatrix(self):
        """This method will display the RGA matrix in a colour block."""
        
        plt.figure("Relative Gain Array")
        [r, c] = self.bristol.bristolmatrix.shape
        plt.imshow(self.bristol.bristolmatrix, cmap=plt.cm.gray_r, interpolation='nearest', extent=[0, 1, 0, 1])
        lenofinputs = len(self.listofinputs)
        outputs = self.bristol.vars[lenofinputs:]
        rstart = 1.0 / (2.0 * r)
        cstart = 1.0 / (2.0 * c)
        rincr = 1.0 / r
        cincr = 1.0 / c
        revinputs = []
        revinputs.extend(self.listofinputs)
        revinputs.reverse()
        plt.yticks(arange(rstart, 1, rincr), revinputs, fontsize=10)
        plt.xticks(arange(cstart, 1, cincr), outputs, rotation= -45, fontsize=10)
        
        rowstart = (r - 1) * rincr + rstart
        for i in range(r):
            ypos = rowstart - i * rincr
            for j in range(c):
                xpos = cstart + cincr * j - 0.15 * cincr
                val = round(self.bristol.bristolmatrix[i, j], 3)
                if val <= 0.5:
                    colour = 'k'
                else:
                    colour = 'w'
                plt.text(xpos, ypos, val, color=colour, fontsize=10)
   
    def showAll(self):
        """This method is called at the end of the visualisation routine so that
        the user may see the whole collection of figures for the system under
        consideration."""
        
        plt.show()
        
    def normaliseMatrix(self, inputmatrix):
        """This method normalises the absolute value of the input matrix
        in the columns i.e. all columns will sum to 1
        
        It also appears in localGainCalculator but not for long! Unless I forget
        about it..."""
        
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
    
    def displayEigenRankLGf(self, nodepos=None):
        """This method constructs a network graph showing connections and rankings
        in terms of node size going FORWARD and using the local gains. 
        
        It has an optional parameter nodepos which sets the positions of the nodes,
        if left out the node layout defaults to circular. """
        
        
        self.LGF = nx.DiGraph()
        for i in range(self.forwardgain.n):
            for j in range(self.forwardgain.n):
                if (self.forwardgain.gMatrix[i, j] != 0):
                    self.LGF.add_edge(self.forwardgain.gVariables[j], self.forwardgain.gVariables[i]) #draws the connectivity graph to visualise rankArray
         
         
        plt.figure("Node Rankings: Local Gain Forward: Scaled")
        rearrange = self.LGF.nodes()

        for node in self.LGF.nodes():
            self.LGF.add_node(node, importance=self.forwardgain.rankDict[node])
        
        nodelabels = dict((n, [n, round(self.forwardgain.rankDict[n], 3)]) for n in self.LGF.nodes())
        sizeArray = [self.forwardgain.rankDict[var] * 10000 for var in rearrange]
        
        if nodepos == None:
            nodepos = nx.circular_layout(self.LGF)        
        
        nx.draw_networkx(self.LGF, pos=nodepos , labels=nodelabels, node_size=sizeArray, node_color='y')
        nx.draw_networkx_edges(self.LGF, pos=nodepos)
        plt.axis("off")

    def displayEigenRankGGf(self, nodepos=None):
        """This method constructs a network graph showing connections and rankings
        in terms of node size going FORWARD and using unity gains between all node
        i.e. the google rank. 
        
        It has an optional parameter nodepos which sets the positions of the nodes,
        if left out the node layout defaults to circular. """
    
        self.GGF = nx.DiGraph()
        for i in range(self.gfgain.n):
            for j in range(self.gfgain.n):
                if (self.gfgain.gMatrix[i, j] != 0):
                    self.GGF.add_edge(self.gfgain.gVariables[j], self.gfgain.gVariables[i])
         
         
        plt.figure("Node Rankings: Google Gain Forward: Scaled")
        rearrange = self.GGF.nodes()
        
        for node in self.GGF.nodes():
            self.GGF.add_node(node, importance=self.gfgain.rankDict[node])
        
        nodelabels = dict((n, [n, round(self.gfgain.rankDict[n], 3)]) for n in self.GGF.nodes())
        sizeArray = [self.gfgain.rankDict[var] * 10000 for var in rearrange]
        
        if nodepos == None:
            nodepos = nx.circular_layout(self.GGF)           
        
        nx.draw_networkx(self.GGF, pos=nodepos , labels=nodelabels, node_size=sizeArray, node_color='y')
        nx.draw_networkx_edges(self.GGF, pos=nodepos)
        plt.axis("off")
        
    def displayEigenRankGGb(self, nodepos=None):
        """This method constructs a network graph showing connections and rankings
        in terms of node size going BACKWARD and using unity gains between all node
        i.e. the google rank. 
        
        It has an optional parameter nodepos which sets the positions of the nodes,
        if left out the node layout defaults to circular. """
        
        
        self.GGB = nx.DiGraph()
        for i in range(self.gbgain.n):
            for j in range(self.gbgain.n):
                if (self.gbgain.gMatrix[i, j] != 0):
                    self.GGB.add_edge(self.gbgain.gVariables[j], self.gbgain.gVariables[i])
         
         
        plt.figure("Node Rankings: Google Gain Backward: Scaled")
        rearrange = self.GGB.nodes()
    
        for node in self.GGB.nodes():
            self.GGB.add_node(node, importance=self.gbgain.rankDict[node])
            
        nodelabels = dict((n, [n, round(self.gbgain.rankDict[n], 3)]) for n in self.GGB.nodes())
        sizeArray = [self.gbgain.rankDict[var] * 10000 for var in rearrange]
        
        if nodepos == None:
            nodepos = nx.circular_layout(self.GGB)           
        
        nx.draw_networkx(self.GGB, pos=nodepos , labels=nodelabels, node_size=sizeArray, node_color='y')
        nx.draw_networkx_edges(self.GGB, pos=nodepos)
        plt.axis("off")

    def displayEigenRankLGb(self, nodepos=None):
        """This method constructs a network graph showing connections and rankings
        in terms of node size going BACKWARD and using the local gains. 
        
        It has an optional parameter nodepos which sets the positions of the nodes,
        if left out the node layout defaults to circular. """
        
        
        self.LGB = nx.DiGraph()
        for i in range(self.backwardgain.n):
            for j in range(self.backwardgain.n):
                if (self.backwardgain.gMatrix[i, j] != 0):
                    self.LGB.add_edge(self.backwardgain.gVariables[j], self.backwardgain.gVariables[i]) #draws the connectivity graph to visualise rankArray
         
         
        plt.figure("Node Rankings: Local Gain Backward: Scaled")
        rearrange = self.LGB.nodes()
        
        for node in self.LGB.nodes():
            self.LGB.add_node(node, importance=self.backwardgain.rankDict[node])
        
        nodelabels = dict((n, [n, round(self.backwardgain.rankDict[n], 3)]) for n in self.LGB.nodes())
        sizeArray = [self.backwardgain.rankDict[var] * 10000 for var in rearrange]
        
        if nodepos == None:
            nodepos = nx.circular_layout(self.LGB)        
        
        nx.draw_networkx(self.LGB, pos=nodepos , labels=nodelabels, node_size=sizeArray, node_color='y')
        nx.draw_networkx_edges(self.LGB, pos=nodepos)
        plt.axis("off")
        
    def displayEigenRankNormalForward(self, nodepos=None):
        """This method constructs a network graph showing connections and rankings
        in terms of node size going FORWARD and using the local gains. (it is not
        scaled) 
        
        It has an optional parameter nodepos which sets the positions of the nodes,
        if left out the node layout defaults to circular. """
        
        
        self.NFG = nx.DiGraph()
        for i in range(self.normalforwardgain.n):
            for j in range(self.normalforwardgain.n):
                if (self.normalforwardgain.gMatrix[i, j] != 0):
                    self.NFG.add_edge(self.normalforwardgain.gVariables[j], self.normalforwardgain.gVariables[i]) #draws the connectivity graph to visualise rankArray
         
         
        plt.figure("Node Rankings: Local Gain Forward: Normal")
        rearrange = self.NFG.nodes()
        
        for node in self.NFG.nodes():
            self.NFG.add_node(node, importance=self.normalforwardgain.rankDict[node])
        
        nodelabels = dict((n, [n, round(self.normalforwardgain.rankDict[n], 3)]) for n in self.NFG.nodes())
        sizeArray = [self.normalforwardgain.rankDict[var] * 10000 for var in rearrange]
        
        if nodepos == None:
            nodepos = nx.circular_layout(self.NFG)        
        
        nx.draw_networkx(self.NFG, pos=nodepos , labels=nodelabels, node_size=sizeArray, node_color='y')
        nx.draw_networkx_edges(self.NFG, pos=nodepos)
        plt.axis("off")        
        
    def displayEigenRankNormalBackward(self, nodepos=None):
        """This method constructs a network graph showing connections and rankings
        in terms of node size going BACKWARD and using the local gains. (it is not
        scaled) 
        
        It has an optional parameter nodepos which sets the positions of the nodes,
        if left out the node layout defaults to circular. """
        
        
        self.NBG = nx.DiGraph()
        for i in range(self.normalbackwardgain.n):
            for j in range(self.normalbackwardgain.n):
                if (self.normalbackwardgain.gMatrix[i, j] != 0):
                    self.NBG.add_edge(self.normalbackwardgain.gVariables[j], self.normalbackwardgain.gVariables[i]) #draws the connectivity graph to visualise rankArray
         
         
        plt.figure("Node Rankings: Local Gain Backward: Normal")
        rearrange = self.NBG.nodes()
        
        for node in self.NBG.nodes():
            self.NBG.add_node(node, importance=self.normalbackwardgain.rankDict[node])
        
        nodelabels = dict((n, [n, round(self.normalbackwardgain.rankDict[n], 3)]) for n in self.NBG.nodes())
        sizeArray = [self.normalbackwardgain.rankDict[var] * 10000 for var in rearrange]
        
        if nodepos == None:
            nodepos = nx.circular_layout(self.NBG)        
        
        nx.draw_networkx(self.NBG, pos=nodepos , labels=nodelabels, node_size=sizeArray, node_color='y')
        nx.draw_networkx_edges(self.NBG, pos=nodepos)
        plt.axis("off") 
        
    def displayEigenRankBlend(self, nodummyvariablelist, alpha, nodepos=None):
        """This method displays the blended weightings of nodes i.e. it takes
        both forward and backward rankings into account.
        
        Note that this is purely ranking i.e. the standard google rankings do
        not come into play yet."""
        
        self.blendedranking = dict()
        for variable in nodummyvariablelist:
            self.blendedranking[variable] = (1 - alpha) * self.forwardgain.rankDict[variable] + (alpha) * self.backwardgain.rankDict[variable]
            
        
        self.EBG = nx.DiGraph()
        for i in range(self.normalforwardgain.n):
            for j in range(self.normalforwardgain.n):
                if (self.normalforwardgain.gMatrix[i, j] != 0):
                    self.EBG.add_edge(self.normalforwardgain.gVariables[j], self.normalforwardgain.gVariables[i]) #draws the connectivity graph to visualise rankArray
         
         
        plt.figure("Blended Node Rankings")
        rearrange = self.EBG.nodes()
        
        for node in self.EBG.nodes():
            self.EBG.add_node(node, importance=self.blendedranking[node])
        
        nodelabels = dict((n, [n, round(self.blendedranking[n], 3)]) for n in self.EBG.nodes())
        sizeArray = [self.blendedranking[var] * 10000 for var in rearrange]
        
        if nodepos == None:
            nodepos = nx.circular_layout(self.EBG)        
        
        nx.draw_networkx(self.EBG, pos=nodepos , labels=nodelabels, node_size=sizeArray, node_color='y')
        nx.draw_networkx_edges(self.EBG, pos=nodepos)
        plt.axis("off")
        
        tt = sorted(self.blendedranking.iteritems(), key = itemgetter(1), reverse=True)
        for ttt in tt:
            print(ttt)
        
    def displayEigenRankBlendGoogle(self, nodummyvariablelist, alpha, nodepos=None):
        """This method displays the blended weightings of nodes i.e. it takes
        both forward and backward rankings into account.
        
        Note that this is purely ranking i.e. the standard google rankings do
        not come into play yet."""
        
        self.blendedrankingGoogle = dict()
        for variable in nodummyvariablelist:
            self.blendedrankingGoogle[variable] = (1 - alpha) * self.gfgain.rankDict[variable] + (alpha) * self.gbgain.rankDict[variable]
            
        
        self.EBGG = nx.DiGraph()
        for i in range(self.normalforwardgain.n):
            for j in range(self.normalforwardgain.n):
                if (self.normalforwardgain.gMatrix[i, j] != 0):
                    self.EBGG.add_edge(self.normalforwardgain.gVariables[j], self.normalforwardgain.gVariables[i]) #draws the connectivity graph to visualise rankArray
         
         
        plt.figure("Blended Node Rankings: Google")
        rearrange = self.EBGG.nodes()
        
        for node in self.EBGG.nodes():
            self.EBGG.add_node(node, importance=self.blendedrankingGoogle[node])
        
        nodelabels = dict((n, [n, round(self.blendedrankingGoogle[n], 3)]) for n in self.EBGG.nodes())
        sizeArray = [self.blendedrankingGoogle[var] * 10000 for var in rearrange]
        
        if nodepos == None:
            nodepos = nx.circular_layout(self.EBGG)        
        
        nx.draw_networkx(self.EBGG, pos=nodepos , labels=nodelabels, node_size=sizeArray, node_color='y')
        nx.draw_networkx_edges(self.EBGG, pos=nodepos)
        plt.axis("off")         
        
    def displayEdgeWeights(self, nodepos=None):
        """This method will compute and store the edge weights of the ranking web.
        
        It *NEEDS* the methods displayEigenBlend and displayEigenBlendGoogle to have
        been run!"""      
                
        self.P = nx.DiGraph()
        self.edgelabels = dict()
        
        for i in range(self.normalforwardgain.n):
            for j in range(self.normalforwardgain.n):
                if (self.normalforwardgain.gMatrix[i, j] != 0):
                    temp = self.normalforwardgain.gMatrix[i, j] * self.blendedranking[self.normalforwardgain.gVariables[j]] - self.blendedrankingGoogle[self.normalforwardgain.gVariables[j]] * self.normalforwardgoogle.gMatrix[i, j]
                    
                    self.edgelabels[(self.normalforwardgain.gVariables[j], self.normalforwardgain.gVariables[i])] = round(-1 * temp, 4)
                    self.P.add_edge(self.normalforwardgain.gVariables[j], self.normalforwardgain.gVariables[i], weight= -1 * temp)
         
        plt.figure("Edge Weight Graph")
        
        for node in self.P.nodes():
            self.P.add_node(node, importanceNormal=self.blendedranking[node])
            self.P.add_node(node, importanceGoogle=self.blendedrankingGoogle[node])
        
        if nodepos == None:
            nodepos = nx.circular_layout(self.P)        
        
        nx.draw_networkx(self.P, pos=nodepos)
        nx.draw_networkx_edge_labels(self.P, pos=nodepos, edge_labels=self.edgelabels, label_pos=0.3)
        nx.draw_networkx_edges(self.P, pos=nodepos, width=2.5, edge_color='k', style='solid', alpha=0.15)
        nx.draw_networkx_nodes(self.P, pos=nodepos, node_color='y', node_size=450)        
        plt.axis("off") 
        
    def createPairingDict(self, variablestocontrol=None):
        """This method should create a dictionary with every pairing as a distinct key
        and the min path edge weight sum the value.
        
        This method requires dispEigenWeightsBlend etc..."""
        #recursive method to return all the possible paths by traveling to a node
        #only once
        def getAllTours(graph, startnode, endnode, path=[]):
            path = path + [startnode]
            if startnode == endnode:
                return [path]
            if startnode not in nx.nodes(graph):
                return []
            paths = []
            for node in nx.neighbors(graph, startnode):
                if node not in path:
                    newpaths = getAllTours(graph, node, endnode, path)
                    for newpath in newpaths:
                        paths.append(newpath)
                        
            return paths
        
        #this sub-method will calculate the minimum path length between 2 nodes
        
        def calculateMinTour(graph, inputnode, outputnode):
            listofpossibletours = getAllTours(graph, inputnode, outputnode)
            minweight = float('inf')    
            for possibility in listofpossibletours:
                pathweight = 0
                for node in range(len(possibility) - 1):
                    pathweight = pathweight + graph[possibility[node]][possibility[node + 1]]['weight']
                if pathweight < minweight:
                    minweight = pathweight
            return minweight   
   
        if variablestocontrol == None:
            controlme = self.listofoutputs
        else:
            controlme = variablestocontrol

        
        self.pathlengthsdict = dict()
        for x in self.listofinputs:
            for y in controlme:
                self.pathlengthsdict[(x, y)] = calculateMinTour(self.P, x, y)

    def calculateAndDisplayBestControl(self, variablestocontrol=None, nodepositions=None, permute=False):
        """This method should calculate the best possible control settings i.e.
        which variables to pair with which other variables.
        The default tries to control the most important variables according to
        the ranking algorithm.
        Needs dispEigenBlend and calculateEdgeWeights. 
        
        ASSUME: You will always have as many or more controlled variables
        as manipulated variables!!!  (The the code won't work properly otherwise...) 
        
        For large systems a much more memory efficient system needs to be designed. To 
        this end the default parameter permute will ensure that you use a greedy
        approach to determine pairings unless it is set to True. This greedy approach 
        takes about 2 min in the Tennessee Eastman problem. """
        
        self.createPairingDict(variablestocontrol)   
        print("Pair Dictionary Created")        
        
        if variablestocontrol == None:
            controlme = self.listofoutputs
        else:
            controlme = variablestocontrol
            
        """Unfortunately, this method is not suitable for large systems. """ 
        if permute:           
            #calculate all control permutations
            controllers = self.listofinputs
            r = len(controllers)
            sequence = permutations(controlme, r)
            prevbestconfig = []
            prevrowsum = float('inf')  
            rowsum = 0
            print("Start Itertions: All Permutations") #for your peace of mind
            for x in sequence:
                possiblepairing = []
                for y in izip(controllers, x):
                    possiblepairing.append(y)
                    rowsum = rowsum + self.pathlengthsdict[y]
                    if (rowsum == float('inf')):
                        break
                if rowsum < prevrowsum:
                    prevbestconfig = possiblepairing
                    prevrowsum = rowsum
                rowsum = 0
        else:
            
            """A more and less reasonable method to determine best pairs"""
            prevbestconfig = []
            rankingsdesc = [x[0] for x in sorted(self.blendedranking.iteritems(), key=itemgetter(1), reverse=True)]
            controldesc = [y for y in rankingsdesc if y in controlme]
            print("Start Iterations: Greedy Pairing")
            """Fixed method: will attempt to force every MV to a unique CV"""
            usedMV = []
            for x in controldesc:
                MV = None
                previter = float('inf')
                for y in self.listofinputs:
                    index = (y, x)
                    if y not in usedMV:
                        if previter > self.pathlengthsdict[index]:
                            recpair = index
                            MV = y
                            previter = self.pathlengthsdict[index]
                    
                    
                usedMV.append(MV)
                prevbestconfig.append(recpair)   
               
        print("The recommended control pairs")
        for x in prevbestconfig:
            print(x)
        
        self.F = nx.DiGraph()
        self.F = self.G.copy() #remember G is the basis graph
        self.F.add_edges_from(self.F.edges(), edgecolour='k')
        
        pairlist = []
        for element in prevbestconfig:
            pairlist.append((element[1], element[0]))
            self.F.add_edge(element[1], element[0], edgecolour='r')
          
        edgecolorlist = ["r" if element in pairlist else "k" for element in self.F.edges()]
        
        if nodepositions == None:
            nodepositions = nx.circular_layout(self.G)
        
        plt.figure("Best Controller Pairs: Eigenvector Approach")            
        nx.draw_networkx(self.F, pos=nodepositions)
        nx.draw_networkx_edges(self.F, pos=nodepositions, width=2.5, edge_color=edgecolorlist, style='solid', alpha=0.15)
        nx.draw_networkx_nodes(self.F, pos=nodepositions, node_color='y', node_size=450)
        plt.axis('off')
        
    def exportToGML(self):
        """This method serves to export all the graphs created to GML files. It detects which 
        objects have been created."""
        
        try:
            if self.G:
                print("G exists")
                nx.write_gml(self.G, "graphG.gml")
        except:
            print("G does not exist")
        
        try:
            if self.EBG:
                print("EBG exists")
                nx.write_gml(self.EBG, "graphEBG.gml")
        except:
            print("EBG does not exist")
            
        try:
            if self.EBGG:
                print("EBGG exists")
                nx.write_gml(self.EBGG, "graphEBGG.gml")
        except:
            print("EBGG does not exist")
            
        try:
            if self.F:
                print("F exists")
                nx.write_gml(self.F, "graphF.gml")
        except:
            print("F does not exist")
        
        try:
            if self.P:
                print("P exists")
                nx.write_gml(self.P, "graphP.gml")
        except:
            print("P does not exist")
            
        try:
            if self.G1:
                print("G1 exists")
                nx.write_gml(self.G1, "graphG1.gml")
        except:
            print("G1 does not exist")
            
        try:
            if self.G2:
                print("G2 exists")
                nx.write_gml(self.G2, "graphG2.gml")
        except:
            print("G2 does not exist")    
            
        try:
            if self.GGF:
                print("GGF exists")
                nx.write_gml(self.GGF, "graphGGF.gml")
        except:
            print("GGF does not exist")
            
        try:
            if self.GGB:
                print("GGB exists")
                nx.write_gml(self.GGB, "graphGGB.gml")
        except:
            print("GGB does not exist")
            
        try:
            if self.LGB:
                print("LGB exists")
                nx.write_gml(self.LGB, "graphLGB.gml")
        except:
            print("LBG does not exist")
            
        try:
            if self.LGF:
                print("LGF exists")
                nx.write_gml(self.LGF, "graphLGF.gml")
        except:
            print("LGF does not exist")
            
        try:
            if self.NFG:
                print("NFG exists")
                nx.write_gml(self.NFG, "graphNFG.gml")
        except:
            print("NFG does not exist")
            
        try:
            if self.NBG:
                print("NBG exists")
                nx.write_gml(self.NBG, "graphNBG.gml")
        except:
            print("NBG does not exist")
