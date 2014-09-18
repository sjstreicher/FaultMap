# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 15:01:12 2014

@author: s13071832
"""

import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq

time   = np.linspace(0,10,2000)
signal = np.cos(5*np.pi*time) + np.cos(7*np.pi*time) + np.cos(3*np.pi*time)

W = fftfreq(signal.size, d=time[1]-time[0])
f_signal = rfft(signal)

# If our original signal time was in seconds, this is now in Hz
cut_f_signal = f_signal.copy()
cut_f_signal[(W>6)] = 0
cut_f_signal[(W<4)] = 0

cut_signal = irfft(cut_f_signal)

import pylab as plt
plt.subplot(221)
plt.plot(time,signal)
plt.subplot(222)
plt.plot(W,f_signal)
plt.xlim(0,10)
plt.subplot(223)
plt.plot(W,cut_f_signal)
plt.xlim(0,10)
plt.subplot(224)
plt.plot(time,cut_signal)
plt.show()