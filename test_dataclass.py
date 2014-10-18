# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 19:21:26 2014

@author: Simon
"""

import ranking.gaincalc

mode = 'plants'
case = 'propylene_compressor'
scenario = 'raw_set3'
single_entropies = True


testdataclass = ranking.gaincalc.WeightcalcData(mode, case, single_entropies)
testscenario = testdataclass.scenariodata(scenario)
