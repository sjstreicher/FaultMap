# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 19:21:26 2014

@author: Simon
"""

import ranking.gaincalc

mode = 'plants'
case = 'propylene_compressor'
scenario = 'raw_full'

testdataclass = ranking.gaincalc.WeightcalcData(mode, case)
testscenario = testdataclass.scenariodata(scenario)
