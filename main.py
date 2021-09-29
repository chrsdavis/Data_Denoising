import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']= [8, 4]
plt.rcParams.update({'font.size': 9})

from scipy.fft import rfft, rfftfreq, irfft
import numpy as np


dt = 0.001 # sample rate
t = np.arange(0,1,dt) # duration of sampling
f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t) #array
f_original = f
f = f + 2.5*np.random.randn(len(t)) # random noisy signal
f_noisy = f

# Number of samples
n = len(t)

fhat = rfft(f) # complex array of freq domain
freq = rfftfreq(n, 1 / dt) # arr
power_spectral_density = fhat * np.conj(fhat) / n # remove complex numbers

PSDfilter = power_spectral_density > 100 # create bool arr of non-noisy idx's
PSD = power_spectral_density * PSDfilter
temp = fhat # temp var for graphing
fhat = PSDfilter * fhat

fclean = irfft(fhat)

plt.plot(t,f_noisy) # original 
plt.show()

plt.plot(freq,np.abs(temp)) # original freq domain
plt.show()

plt.plot(freq,np.abs(fhat)) # filtered freq domain
plt.show()

plt.plot(t,fclean) # filtered signal
plt.show()
