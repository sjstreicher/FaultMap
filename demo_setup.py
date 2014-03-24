# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 21:00:03 2014

@author: Simon
"""

import json
import os


def runsetup():
    # Load directories config file
    dirs = json.load(open('config.json'))
    # Get data and preferred export directories from directories config file
    dataloc = os.path.expanduser(dirs['dataloc'])
    saveloc = os.path.expanduser(dirs['saveloc'])
    infodynamicsloc = os.path.expanduser(dirs['infodynamicsloc'])
    # Define plant and case names to run
    plant = 'tennessee_eastman'
    # Define plant data directory
    plantdir = os.path.join(dataloc, 'plants', plant)
    cases = ['dist11_closedloop', 'dist11_closedloop_pressup', 'dist11_full',
             'presstep_closedloop', 'presstep_full']
    # Load plant config file
    caseconfig = json.load(open(os.path.join(plantdir, plant + '.json')))
    # Get sampling rate
    sampling_rate = caseconfig['sampling_rate']

    # Load directories config file
    dirs = json.load(open('config.json'))
    # Get data and preferred export directories from directories config file
    dataloc = os.path.expanduser(dirs['dataloc'])
    saveloc = os.path.expanduser(dirs['saveloc'])
    # Define plant and case names to run
    plant = 'tennessee_eastman'
    # Define plant data directory
    plantdir = os.path.join(dataloc, 'plants', plant)
    cases = ['dist11_closedloop', 'dist11_closedloop_pressup', 'dist11_full',
             'presstep_closedloop', 'presstep_full']
    # Load plant config file
    caseconfig = json.load(open(os.path.join(plantdir, plant + '.json')))
    # Get sampling rate
    sampling_rate = caseconfig['sampling_rate']

    return cases, saveloc, caseconfig, plantdir, sampling_rate, infodynamicsloc
