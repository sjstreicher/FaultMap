"""This script runs all other scripts required for calculating a ranking

@author: St. Elmo Wilken, Simon Streicher

"""

from ranking.controlranking import LoopRanking
from ranking.formatmatrices import FormatMatrix

import json

filesloc = json.load(open('config.json'))

datamatrix = FormatMatrix(filesloc['connections'], filesloc['data'], 0)
controlmatrix = LoopRanking(datamatrix.scaledforwardgain,
                            datamatrix.scaledforwardvariablelist,
                            datamatrix.scaledforwardconnection,
                            datamatrix.scaledbackwardgain,
                            datamatrix.scaledbackwardvariablelist,
                            datamatrix.scaledbackwardconnection,
                            datamatrix.nodummyvariablelist)

#controlmatrix.display_control_importances([], datamatrix.nodummyconnection)
#controlmatrix.show_all()
#controlmatrix.exportogml()