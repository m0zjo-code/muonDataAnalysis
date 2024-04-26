# -*- coding: utf-8 -*-
"""
@author: Adam Rawlinson M0ZJQ adamr.westfield@gmail.com
"""
# OK so this version uses open file dialogue, automatically ignores the rubbish from the file, then anayses rate. bins date and plots 
# The code now also writes the binned and plotted output to a text file. Care as it overwrites the binned output file each time 
# I have used field numbers then added a name - as the headers are hard to obtain from the text file.
# This version also now has a input dialogue box for binning. And I have added a default binning value
# and pretty colours on the plots

import os
import tkinter as tk
from tkinter import filedialog

def open_file():
    root = tk.Tk()
    root.withdraw()  # I hide the main window
    filename = filedialog.askopenfilename()  # I open the file dialog
    return filename  # Returns the filename

# assign a name to the file then I use it in code 
muondata = open_file()
print(muondata)
# These lines fire up the plotting and analysis modules after reading the file (before any other processing)
from scipy.fft import fft
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
#import tkinter as tk

# This started as experimental code for binning input but seems to work really well. Careful on granularity 
from tkinter import simpledialog

# Set an initial default value
bin_value = 500.0
# Sets the binning value. Note to self: careful with values larger than 1000 as can lose fidelity 
#bin_value = 200 works well for upto about 1 week. 1500 for a month

def get_numeric_value():
    global bin_value
    try:
        user_input = simpledialog.askstring("Numeric Input", f"Enter binning value (default: {bin_value}):")
        if user_input:
            bin_value = float(user_input)
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

root = tk.Tk()
root.withdraw()  # Hide the main window

get_numeric_value()

# Now I use the 'bin_value' variable in the code.
print(f"Assigned input to 'bin_value': {bin_value}")

# this is the end of the experimental binning code


# Read the file
df = pd.read_csv(muondata, delim_whitespace=True, skiprows=30, header=None, usecols=[0, 1, 2, 3, 4], encoding='ISO-8859-1')
df.columns = ['Field1', 'Field2', 'Field3', 'Arduinotime', 'Level']# I am adding field names manually as they are embedded
# inside rubbish in the file opening lines.
# Here I am using Date Time and the Event counter to provide a rate only (not intensity)

# Combine the date from 'comp_date' and time from 'Comp_time'
df['Datetime'] = pd.to_datetime(df['Field1'] + ' ' + df['Field2'], format = 'mixed')

# Calculate the time difference
df['time_diff'] = df['Datetime'].diff().dt.total_seconds()

# Calculate the counter difference
df['counter_diff'] = df['Field3'].diff()

# Calculate the rate
df['event_rate'] = df['counter_diff'] / df['time_diff']

# Bin the event rate over bin_value seconds
df['time_bin'] = (df['Datetime'].view(np.int64) // 10**9) // bin_value

df_binned = df.groupby('time_bin').mean(numeric_only=True)  # Explicitly set numeric_only=True


df_binned['Datetime'] = pd.to_datetime(df_binned.index * bin_value, unit='s')

# Calculate the average event rate per second
df_binned['event_rate_avg'] = df_binned['event_rate'] / 1
#The divisor here is experimental. I think the binning strategy above then distorts the average rates.
#A value of 3 shows similar rates to device display to counter cumulative rate across three stages of calc.

#subplots code
fig, ((ax1, ax2), (ax3, ax4))=plt.subplots(2,2, facecolor="lightblue", figsize=(12,8),layout="constrained")
fig.suptitle('M0ZJQ - Muon Detector Analysis', fontsize=12)

#Plotting the RATE OF DETECTIONS
ax1.plot(df_binned['Datetime'], df_binned['event_rate_avg'], label='Muon Event Rate', color='r')
ax1.set_ylabel('Average Event Rate')
#ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=2500))  # interval can be adjusted to make look pretty 
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))#%Y-:%S.%f add these back in if I want sub-second plots
ax1.set_title(f'Muon Event Rate (Binning Value: {bin_value} event:seconds)')
ax1.grid(True, color="blue", which='both', linestyle='--', linewidth=0.5)

#Plotting the intensity
ax2.plot(df_binned['Datetime'], df_binned['Level'], label='Muon Intensity (0-1023)', color='b')
ax2.set_ylabel('ADU intensity')
ax2.set_title(f'Muon Intensity (Binning Value: {bin_value} event:seconds)')
ax2.grid(True, color="blue", which='both', linestyle='--', linewidth=0.5)

# Get the directory and base name of the input file
input_directory, input_filename = os.path.split(muondata)
input_basename = os.path.splitext(input_filename)[0]  # Remove the extension
# really nifty bit to save the plot to a file same folder as the input file with name as suffix
output_file = os.path.join(input_directory, f"output_{input_basename}.txt")
with open(output_file, "w") as f:
    for xi, yi in zip(df_binned['Datetime'], df_binned['event_rate_avg']):
        f.write(f"{xi.strftime('%Y-%m-%d %H:%M:%S')}\t{yi:.6f}\n")  # Format the datetime
# Echo to the console that a job is well done 
print(f"Output plot saved to {output_file} as a plain text file")



# so im using the binned output (same as used on graph) to try and do a long time FFT on the results
signal=df_binned['event_rate_avg'].values
signal2=df_binned['Level'].values
# Compute the FFT - didnt realise matoplotlib did this !
fft_result = fft(signal)
fft_result2 = fft(signal2)

# Calculate the corresponding frequencies
sampling_rate = 1  # Adjust if needed but increased peak as well!!!!!
frequencies = np.fft.fftfreq(len(signal), d=1/sampling_rate)
frequencies2 = np.fft.fftfreq(len(signal2), d=1/sampling_rate)

# Plot the frequency spectrum - time 
#plt.figure(figsize=(12, 6), facecolor='lightskyblue')
ax3.plot(frequencies, np.abs(fft_result), 'orange')
ax3.set_xlabel('Time domain')
ax3.set_ylabel('Rate Value')
ax3.set_title('FFT Analysis of rate data')
ax3.set_yscale('log')  # Set the y-axis to log scale
ax3.grid(True, color="blue", linestyle='--', linewidth=0.2)

#plt.show()

# Plot the frequency spectrum - level
#plt.figure(figsize=(12, 6), facecolor='lightgreen')
ax4.plot(frequencies2, np.abs(fft_result2), 'green')
ax4.set_xlabel('Time domain')
ax4.set_ylabel('ADU Level')
ax4.set_title('FFT Analysis of intensity data')
ax4.set_yscale('log')# Set the y-axis to log scale
ax4.grid(True, color="blue", linestyle='--', linewidth=0.2)

plt.show()

#Plot a histogram of levels in sample period 
plt.figure(figsize=(12, 6), facecolor='orange')
plt.hist(df_binned['event_rate_avg'], bins=int((bin_value/10)))
plt.xlabel('Rate')
plt.ylabel('Occurences')
plt.title(f'Histogram - Binning: {bin_value/10}', fontsize=14)
plt.show()
