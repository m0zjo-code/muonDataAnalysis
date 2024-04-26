# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:21:49 2024

@author: M0ZJO
"""
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from scipy.fft import fftfreq
from scipy import signal

# read text file into pandas DataFrame
#df = pd.read_csv("Muondata-WF-090324-sanitised.zip", sep=" ")
df = pd.read_csv("muon-counts-160424-sanitised.zip", sep=" ")

# display DataFrame
#print(df)

# Firstly - we need to calculate a regulary sampled event rate dataset. 
# Bin length - in seconds
bin_len = "1000s"

# Covert to DateTime (Much easier to work with!)
df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format='mixed')
df["Epoch_ns"] = df["DateTime"].astype('int64')

# Map in required metric (Just use "1" if only the event rate is of interest)
df["Event_Rate"] = df["Meta0"]

# Bin the dataset
# Note that the mean() of the aggregate event power is used here.
# Sum() should be used if the total dose is of interest
workingSet = df.resample(bin_len, on='DateTime').Event_Rate.mean().to_frame()

#  Plot the binned dataset
plt.figure(1)
plt.plot(workingSet["Event_Rate"])
plt.title("Raw Binned Muon Detector Rates")
plt.xlabel("DateTime")
plt.ylabel("Detections/min")
plt.grid(which='both', axis='both')
plt.show()

# Now - we need to debias this dataset for period-analysis
workingSet["Event_Rate_Debiased"] = workingSet["Event_Rate"] - workingSet["Event_Rate"].mean()

#  Plot the debiased dataset
plt.figure(2)
plt.plot(workingSet["Event_Rate_Debiased"])
plt.title("Raw Binned Muon Detector Rates (Debiased)")
plt.xlabel("DateTime")
plt.ylabel("Detections/min - Zero Mean")
plt.grid(which='both', axis='both')
#plt.ylim([-100, +100])
plt.show()

# Compute FFT of Data - Before any filtering
fft_val = np.fft.fft(workingSet["Event_Rate"])
fft_pow = (fft_val.real ** 2) + (fft_val.imag ** 2)
fft_freqs = fftfreq(workingSet["Event_Rate"].size, d = 120)
fft_pos_len = int((workingSet["Event_Rate"].size/2)-1)

plt.figure(2)
plt.semilogy(fft_freqs[0:fft_pos_len], fft_pow[0:fft_pos_len])
plt.title("Muon Detector Rates (FFT)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.grid(which='both', axis='both')
plt.show()

# Compute PWelch of Zero Mean Data
f, Pxx_den = signal.welch(workingSet["Event_Rate_Debiased"], 1/1000, nperseg=256)
plt.figure(3)
plt.title("PWelch PSD Estimate of Series")
plt.plot(f, Pxx_den)
#plt.semilogy(f, Pxx_den)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(which='both', axis='both')
plt.show()


