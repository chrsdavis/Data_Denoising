import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']= [8, 4]
plt.rcParams.update({'font.size': 9})

from scipy.fft import rfft, rfftfreq, irfft
import numpy as np


def clean_data(f_noisy,n,dt,lim):
  fhat = rfft(f_noisy) # complex array
  power_spectral_density = fhat * np.conj(fhat) / n # arr
  PSDfilter = power_spectral_density > lim # create bool arr of non-noisy idx's
  fhat = PSDfilter * fhat
  return irfft(fhat)

def gen_signal_clean(dt):
  t = np.arange(0,1,dt)
  f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t) #array
  return f

def gen_signal_noisy(dt):
  t = np.arange(0,1,dt)
  f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t) #array
  f_noisy = f + 2.5*np.random.randn(len(t))
  return f_noisy

def plot_noisy(dt):
  t = np.arange(0,1,dt)
  f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t) #array
  f_noisy = f + 2.5*np.random.randn(len(t))
  plt.plot(t,f_noisy)
  plt.show()

def plot_freq_domain(f_noisy,n,dt):
  fhat = rfft(f_noisy) # complex array
  freq = rfftfreq(n, dt) # arr of bin centers
  plt.plot(freq,np.abs(fhat)) # original freq domain
  plt.show()

def plot_freq_domain_filt(f_noisy,n,dt,lim):
  fhat = rfft(f_noisy) # complex array
  freq = rfftfreq(n, dt) # arr of bin centers
  power_spectral_density = fhat * np.conj(fhat) / n # arr
  PSDfilter = power_spectral_density > lim # create bool arr of non-noisy idx's
  fhat = PSDfilter * fhat
  plt.plot(freq,np.abs(fhat)) # filtered freq domain
  plt.show()

def plot_clean_signal(f_noisy,n,dt,lim):
  fhat = rfft(f_noisy) # complex array
  power_spectral_density = fhat * np.conj(fhat) / n # arr
  PSDfilter = power_spectral_density > lim # create bool arr of non-noisy idx's
  fhat = PSDfilter * fhat
  fclean = irfft(fhat)
  plt.plot(fclean) # filtered signal
  plt.show()
