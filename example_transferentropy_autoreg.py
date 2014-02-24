"""Verifies the working of the transfer entropy calculation code by means of
an example on autoregressive data with known time delay.

Created on Mon Feb 24 14:56:25 2014

@author: Simon Streicher
"""

import transentropy

TRANSENTROPY1 = transentropy.calculate_te(4, 4, 3000, 2000, 10000)
