import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']= [8, 4]
plt.rcParams.update({'font.size': 9})

from scipy.fft import rfft, rfftfreq, irfft
import numpy as np


def denoise(f_noisy, freq_partition = 100, min_power = 0):
  if freq_partition > 100:
    freq_partition = 100
  elif freq_partition < 0:
    freq_partition = 0

  n = len(f_noisy) # Number of samples
  f_trans = rfft(f_noisy) # Complex array of freq domain

  # Zero the bins of high freqs
  f_trans[:int(freq_partition/100 * n)] = 0

  if min_power > 0:
    # Convert complex array to real PSD array
    power_spectral_density = f_trans * np.conj(f_trans) / n
    # Generate filter of insignificant frequencies (>min power)
    PSDfilter = power_spectral_density > min_power # bool arr
    # Zero out bins of frequencies > lim
    f_trans = PSDfilter * f_trans

  # Inverse FFT of filtered signal
  return irfft(f_trans)


def csv_to_arr(csv_filename):
  return np.floor(np.genfromtxt(csv_filename, delimiter=',')[1:,1:-1].flatten()) #[:500]


def diffABS(data_a,data_b,period=5):
  
  # Resize arr to cleanly fit the period size, make all value 0.
  avg = np.zeros(data_a.size - data_a.size % period if data_a.size > data_b.size else data_b.size - data_b.size % period)

  i = 0
  j = period-1 if period-1>0 else 4
  while j<data_b.size-1:
    avg[i:j] = np.abs(data_a[i] * data_b[i:j] - data_b[i] * data_a[i:j])
    i += period
    j += period
  
  return avg


def diff(data_a,data_b,period=5):
  
  # Resize arr to cleanly fit the period size, make all value 0.
  avg = np.zeros(data_a.size - data_a.size % period if data_a.size > data_b.size else data_b.size - data_b.size % period)

  i = 0
  j = period-1 if period-1>0 else 4
  while j<data_b.size-1:
    avg[i:j] = data_a[i] * data_b[i:j] - data_b[i] * data_a[i:j]
    i += period
    j += period
  
  return avg
