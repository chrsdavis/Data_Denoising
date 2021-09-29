import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
#import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft
import numpy as np


dt = 0.001
t = np.arange(0,1,dt)
f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t) #array
f_original = f
f = f + 2.5*np.random.randn(len(t))

# Number of samples
n = len(t)

fhat = rfft(f) # complex array
freq = rfftfreq(n, 1 / dt) # arr
power_spectral_density = fhat * np.conj(fhat) / n # arr

PSDfilter = power_spectral_density > 100
PSD = power_spectral_density * PSDfilter
fhat = PSDfilter * fhat

fclean = irfft(fhat)
