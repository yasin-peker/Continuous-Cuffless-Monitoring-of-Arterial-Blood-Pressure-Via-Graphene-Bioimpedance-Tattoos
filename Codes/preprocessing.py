import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter,filtfilt, hilbert, find_peaks, argrelmin
import scipy.signal
from scipy.fft import fft, fftfreq

######################################## Data Types ########################################
# Index(['time', 'BioZ1', 'BioZ2', 'BioZ3', 'BioZ4'], dtype='object')
# BioZ1 (radial artery closer to wrist) [mOhms] -> Sampling Rate = 1250 Hz
# BioZ2 (radial artery closer to the heart) [mOhms] -> Sampling Rate = 1250 Hz
# BioZ3 (ulnar artery closer to wrist) [mOhms] -> Sampling Rate = 1250 Hz
# BioZ4 (ulnar artery closer to the heart) [mOhms] -> Sampling Rate = 1250 Hz

# Index(['time', 'FinapresBP'], dtype='object')
# finapresBP = Continuous BP measurement with Finapres representing brachial BP [mmHg] -> Sampling Rate = 200 Hz

# Index(['time', 'FinapresPPG'], dtype='object')
# finapresPPG = PPG measurement from Finapres on fingertip [mmHg]-> Sampling Rate = 75Hz

# Index(['time', 'PPG'], dtype='object')
# ppg = PPG measurement from fingertip with BioZ XL board -> Sampling Rate = 1250 Hz

######################################## Setup Definitions ########################################
# HGCP (Hand Grip Cold Pressor): the subject performs a handgrip (HG) exercise for 3 minutes, slowly raising their DBP and SBP, then placing their hand into an ice cold water bucket (cold pressor, CP) for 1 minute to ensure that BP first goes even higher, then very slowly decreases over the 4 minute resting period.
# Cycling: the subject is stationed to perform a set of bike cycling treadmill exercises for 4 minutes, with 4 minutes of break for resting in between.
# Valsalva: session with multiple Valsalva maneuvers. Each Valsalva maneuver consists of a subject pinching their nose while trying to breathe out intensely for 20-30 seconds, creating an extensive buildup of inner pressure, both raising BR, then decreasing, and rapidly increasing it once again very rapidly.
# baseline: At the beginning of data collection with no BP change protocol, participants are at rest.
# rest: no BP change protocol, participants are at rest.

######################################## Data Pre-Processing Functions ########################################
######################################## Lowpass Filter  ########################################
def low_pass_filter(signal, cut_off_frequency, order, sampling_rate):
    # Inputs:
    # signal: 1D input signal to be filtered
    # cut_off_frequency: Cut-off frequency of the lowpass filter
    # sampling_rate: Sampling rate of the signal
    # fs = 1250 Hz for BioZ, fs = 200 Hz for finapresBP, fs = 75 Hz for finapresPPG, fs = 1250 Hz for ppg

    # Define the Lowpass filter
    numerator, denominator = butter(order, cut_off_frequency, btype="lowpass",
                                    analog=False, output="ba", fs=sampling_rate)
    filtered_signal = filtfilt(numerator, denominator, signal)

    return filtered_signal

######################################## Bandpass Filter  ########################################
def band_pass_filter(signal, cut_off_frequency1, cut_off_frequency2, order, sampling_rate):
    # Inputs:
    # signal: 1D input signal to be filtered
    # cut_off_frequency1: Lower cut-off frequency of the bandpass filter
    # cut_off_frequency2: Higher cut-off frequency of the bandpass filter
    # sampling_rate: Sampling rate of the signal
    # fs = 1250 Hz for BioZ, fs = 200 Hz for finapresBP, fs = 75 Hz for finapresPPG, fs = 1250 Hz for ppg

    # Define the Bandpass filter
    numerator, denominator = butter(order, [cut_off_frequency1, cut_off_frequency2], btype="bandpass",
                                    analog=False, output="ba", fs=sampling_rate)
    filtered_signal = filtfilt(numerator, denominator, signal)

    return filtered_signal

######################################## Detrend Signal  ########################################
# Remove linear trend along axis from data
def detrend_signal(signal):
    "Input: 1D Signal, Output: Detrended 1D Signal"
    detrended_signal = scipy.signal.detrend(signal)

    return detrended_signal

######################################## Compute FFT  ########################################
def compute_FFT(signal, raw_time,feature_name):
    "Compute the FFT of a signal according to the feature name"
    duration = raw_time.iloc[-1] - raw_time.iloc[0]
    if feature_name == "BioZ1"\
            or feature_name == "BioZ2"\
            or feature_name == "BioZ3"\
            or feature_name == "BioZ4"\
            or feature_name == "PPG":
        sampling_rate = 1250
    elif feature_name == "FinapresBP":
        sampling_rate = 200
    else:
        sampling_rate = 75

    # Number of samples converted into integer
    N = int(sampling_rate * duration)

    # Convert Pandas Series to NumPy array
    y_fft = fft(signal[feature_name].values)
    x_fft = fftfreq(N, 1 / sampling_rate)

    # Keep only the positive frequencies to match the dimension sizes
    x_fft = x_fft[:N//2]
    y_fft = y_fft[:N//2]

    return x_fft, y_fft

######################################## Envelope Signal  ########################################
def envelope_signal(signal):
    "Compute analytic signal using Hilbert transform to extract the amplitude of the signal"
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)

    return amplitude_envelope

######################################## DataFrame Converter  ########################################
def convert_dataframe(signal, feature):
    "Convert numpy array into a dataframe"
    df_signal = pd.DataFrame(signal)
    df_signal = df_signal.rename(columns={0: feature})

    return df_signal

#################################### Feature Extraction Functions ####################################
######################################## Signal Normalizer  ########################################
def normalize_signal(signal):
    "Normalize the signal to be in the range of [0,1]"
    normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    return normalized_signal

######################################## Peak Calculator  ########################################
def calculate_peaks(signal, feature, height=0.2):
    "Calculate peaks of the signal and return indices of peaks"
    signal_peaks, _ = find_peaks(signal[feature], height=height)
    return signal_peaks

######################################## Heart Rate Calculator  ########################################
def calculate_heart_rate(signal_peak, sampling_rate):
    "Calculate heart rate by finding the time differences between peaks and dividing it by sampling rate"
    time_between_peaks = np.diff(signal_peak[0::2]) / sampling_rate
    heart_rate = 60 / np.mean(time_between_peaks)
    return heart_rate

# Read the BioZ signals
df_BioZ = pd.read_csv("Dataset/subject2_day1/setup01_baseline/data_trial01_bioz.csv")

# Read the finapresBP signal
df_finapresBP = pd.read_csv("Dataset/subject2_day1/setup01_baseline/data_trial01_finapresBP.csv")

# Read the finapresPPG signal
df_finapresPPG = pd.read_csv("Dataset/subject2_day1/setup01_baseline/data_trial01_finapresPPG.csv")

# Read the PPG signal
df_PPG = pd.read_csv("Dataset/subject2_day1/setup01_baseline/data_trial01_ppg.csv")

# Detrend the signal since it has a trend and it has not zero mean
BioZ1_detrend = detrend_signal(df_BioZ["BioZ1"])
BioZ2_detrend = detrend_signal(df_BioZ["BioZ2"])
BioZ3_detrend = detrend_signal(df_BioZ["BioZ3"])
BioZ4_detrend = detrend_signal(df_BioZ["BioZ4"])

PPG_detrend = detrend_signal(df_PPG["PPG"])

# Convert detrend arrays into DataFrames
df_BioZ1_detrend = convert_dataframe(BioZ1_detrend, "BioZ1")
df_BioZ2_detrend = convert_dataframe(BioZ2_detrend, "BioZ2")
df_BioZ3_detrend = convert_dataframe(BioZ3_detrend, "BioZ3")
df_BioZ4_detrend = convert_dataframe(BioZ4_detrend, "BioZ4")

df_PPG_detrend = convert_dataframe(PPG_detrend, "PPG")

# Compute FFT and Plot

x_fft, y_fft = compute_FFT(df_PPG_detrend, df_PPG["time"], "PPG")
plt.plot(x_fft, np.abs(y_fft) / 1e4, label="FFT-PPG")
plt.xlabel("Frequency (Hz)")
plt.ylabel("FFT Amplitude |X(freq)|")
plt.title("Fast Fourier Transform of PPG")
plt.grid(True)
plt.xlim(0, 10)
plt.legend()
plt.show()

x_fft, y_fft = compute_FFT(df_BioZ1_detrend, df_PPG["time"], "BioZ1")
plt.plot(x_fft, np.abs(y_fft) / 1e4, label="FFT-BioZ1")
plt.xlabel("Frequency (Hz)")
plt.ylabel("FFT Amplitude |X(freq)|")
plt.title("Fast Fourier Transform of BioZ1")
plt.grid(True)
plt.xlim(0, 10)
plt.legend()
plt.show()

order_BioZ = 2
sampling_rate_BioZ = 1250
cut_off_BioZ = 6

order_PPG = 2
sampling_rate_PPG = 1250
cut_off1_PPG = 0.1 # DC offset removal
cut_off2_PPG = 50 # High frequency noise removal

# Apply Bandpass Filter centered around the driving AC frequency to remove
# the residual DC offset, 60 Hz interference, and high-frequency noise
band_pass_filtered_PPG = band_pass_filter(df_PPG_detrend["PPG"], cut_off1_PPG, cut_off2_PPG,
                                          order_PPG, sampling_rate_PPG)
# Convert filtered array into dataframe
df_band_pass_filtered_PPG = convert_dataframe(band_pass_filtered_PPG, "PPG")

# Apply Lowpass Filter with a cut-off frequency of 6 Hz to the BioZ signals to
# remove the carrier signal distortion and out-of-band noise
low_pass_filtered_BioZ1 = low_pass_filter(df_BioZ1_detrend["BioZ1"], cut_off_BioZ, order_BioZ, sampling_rate_BioZ)
low_pass_filtered_BioZ2 = low_pass_filter(df_BioZ2_detrend["BioZ2"], cut_off_BioZ, order_BioZ, sampling_rate_BioZ)
low_pass_filtered_BioZ3 = low_pass_filter(df_BioZ3_detrend["BioZ3"], cut_off_BioZ, order_BioZ, sampling_rate_BioZ)
low_pass_filtered_BioZ4 = low_pass_filter(df_BioZ4_detrend["BioZ4"], cut_off_BioZ, order_BioZ, sampling_rate_BioZ)

# Convert filtered array into dataframe
df_low_pass_filtered_BioZ1 = convert_dataframe(low_pass_filtered_BioZ1, "BioZ1")
df_low_pass_filtered_BioZ2 = convert_dataframe(low_pass_filtered_BioZ2, "BioZ2")
df_low_pass_filtered_BioZ3 = convert_dataframe(low_pass_filtered_BioZ3, "BioZ3")
df_low_pass_filtered_BioZ4 = convert_dataframe(low_pass_filtered_BioZ4, "BioZ4")

# Envelope signals
enveloped_BioZ1 = envelope_signal(df_low_pass_filtered_BioZ1["BioZ1"])
enveloped_BioZ2 = envelope_signal(df_low_pass_filtered_BioZ2["BioZ2"])
enveloped_BioZ3 = envelope_signal(df_low_pass_filtered_BioZ3["BioZ3"])
enveloped_BioZ4 = envelope_signal(df_low_pass_filtered_BioZ4["BioZ4"])

# Convert filtered array into dataframe
df_enveloped_BioZ1 = convert_dataframe(enveloped_BioZ1, "BioZ1")
df_enveloped_BioZ2 = convert_dataframe(enveloped_BioZ2, "BioZ2")
df_enveloped_BioZ3 = convert_dataframe(enveloped_BioZ3, "BioZ3")
df_enveloped_BioZ4 = convert_dataframe(enveloped_BioZ4, "BioZ4")

# Normalize the PPG signal
normalized_PPG = normalize_signal(df_band_pass_filtered_PPG)
df_normalized_PPG = convert_dataframe(normalized_PPG, "PPG")

# Calculate the peaks of the PPG signal
PPG_peaks = calculate_peaks(df_normalized_PPG, "PPG")

# Calculate Heart Rate
heart_rate = calculate_heart_rate(PPG_peaks, sampling_rate_PPG)

print(f"Heart Rate: {heart_rate:.2f} beats per minute")

############### Peak detection on BioZ signals #################
# Normalize the BioZ1 and BioZ2 signals and detect peaks
normalized_BioZ1 = normalize_signal(df_low_pass_filtered_BioZ1)
normalized_BioZ2 = normalize_signal(df_low_pass_filtered_BioZ2)

# Convert arrays into dataframes
df_normalized_BioZ1 = convert_dataframe(normalized_BioZ1, "BioZ1")
df_normalized_BioZ2 = convert_dataframe(normalized_BioZ2, "BioZ2")

BioZ1_peaks = calculate_peaks(df_normalized_BioZ1, "BioZ1", height=0.75)
BioZ2_peaks = calculate_peaks(df_normalized_BioZ2, "BioZ2", height=0.735)

print("BioZ1_peaks:", len(BioZ1_peaks))
print("BioZ2_peaks:", len(BioZ2_peaks))

# Local min and max calculation
a = np.diff(np.sign(np.diff(df_normalized_BioZ1["BioZ1"]))).nonzero()[0] + 1 # local min+max
b = (np.diff(np.sign(np.diff(df_normalized_BioZ1["BioZ1"]))) > 0).nonzero()[0] + 1 # local min
c = (np.diff(np.sign(np.diff(df_normalized_BioZ1["BioZ1"]))) < 0).nonzero()[0] + 1 # local max

# Display the local minima and maxima of BioZ signal
plt.plot(df_BioZ["time"], df_normalized_BioZ1)
plt.plot(df_BioZ.time[b], df_normalized_BioZ1.BioZ1[b], "o", label="min")
plt.plot(df_BioZ.time[c], df_normalized_BioZ1.BioZ1[c], "o", label="max")
plt.xlabel('Time (min)')
plt.ylabel('Bioimpedance (mohms)')
plt.title('Local Minima and Maxima Detection of BioZ1 Signal')
plt.legend()
plt.show()

# Display the results
plt.plot(df_BioZ["time"], df_normalized_BioZ1, label='BioZ1 Signal')
plt.plot(df_BioZ.time[BioZ1_peaks], df_normalized_BioZ1.BioZ1[BioZ1_peaks], 'ro', label='Peaks')
plt.title('BioZ1 Signal with Peaks')
plt.xlabel('Time (min)')
plt.ylabel('Bioimpedance (mohms)')
plt.legend()
plt.show()

# Display the results
plt.plot(df_BioZ["time"], df_normalized_BioZ2, label='BioZ2 Signal')
plt.plot(df_BioZ.time[BioZ2_peaks], df_normalized_BioZ2.BioZ2[BioZ2_peaks], 'ro', label='Peaks')
plt.title('BioZ2 Signal with Peaks')
plt.xlabel('Time (min)')
plt.ylabel('Bioimpedance (mohms)')
plt.legend()
plt.show()

plt.plot(df_BioZ["time"], df_normalized_BioZ1, label='BioZ1 Signal', color="blue")
plt.plot(df_BioZ["time"], df_normalized_BioZ2, label='BioZ2 Signal', color="red")
plt.title('BioZ1 and BioZ2 Signals')
plt.xlabel('Time (min)')
plt.ylabel('Bioimpedance (mohms)')
plt.legend()
plt.show()

# Calculate Pulse Transit Time (PTT) as the time between two selected peaks
"""selected_peaks = PPG_peaks[:2]
ptt = np.diff(selected_peaks) / sampling_rate_PPG

min_diastolic_pressure = 70
max_systolic_pressure = 110

# Calculate mean arterial pressure (MAP) for the given range
map_min = min_diastolic_pressure + 2/5 * (max_systolic_pressure - min_diastolic_pressure)
map_max = min_diastolic_pressure + 3/5 * (max_systolic_pressure - min_diastolic_pressure)

# Determine calibration parameters
a = 1 / (map_max - map_min)
b = -map_min * a

# Now you have calibrated values for 'a' and 'b'
# You can use them in your calibration equation
map_estimate = a * ptt[0] + b
sys_bp_estimate = map_estimate + 20
dia_bp_estimate = map_estimate - 10


print(f"Pulse Transit Time (PTT): {ptt[0]:.2f} seconds")
# Display the results
print(f"Mean Arterial Pressure (MAP) Estimate: {map_estimate:.2f} mmHg")
print(f"Systolic Blood Pressure Estimate: {sys_bp_estimate:.2f} mmHg")
print(f"Diastolic Blood Pressure Estimate: {dia_bp_estimate:.2f} mmHg")"""

# Display the peak detected PPG results
"""plt.plot(df_PPG["time"], df_normalized_PPG, label='PPG Signal')
plt.plot(df_PPG.time[PPG_peaks], df_normalized_PPG.PPG[PPG_peaks], 'ro', label='Peaks')
plt.title('PPG Signal with Peaks')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()"""


"""print(df_PPG.PPG[PPG_peaks])
plt.figure()
plt.xlabel("Time (min)"); plt.ylabel("PPG (a.u.)")
plt.plot(df_normalized_PPG.PPG[PPG_peaks], "o",label="PPG-peaks", color="red")
plt.plot(df_normalized_PPG, label="PPG-Filtered", color="blue")
plt.legend()
plt.grid(True)
plt.show()"""

# Plot the BioZ vs time
plt.figure(figsize=(8,3))

plt.subplot(221)
plt.xlabel("Time (min)"); plt.ylabel("Bioimpedance (mOhms)")
plt.plot(df_BioZ["time"], df_low_pass_filtered_BioZ1, label="BioZ1-Filtered", color="green")
plt.plot(df_BioZ["time"], df_BioZ["BioZ1"], label="BioZ1-Raw", color="black", alpha=0.5)
plt.legend(loc="upper right")
plt.grid(True)

plt.subplot(222)
plt.xlabel("Time (min)"); plt.ylabel("Bioimpedance (mOhms)")
plt.plot(df_BioZ["time"], df_low_pass_filtered_BioZ2, label="BioZ2-Filtered", color="red")
plt.plot(df_BioZ["time"], df_BioZ["BioZ2"], label="BioZ2-Raw", color="black", alpha=0.5)
plt.legend(loc="lower right")
plt.grid(True)

plt.subplot(223)
plt.xlabel("Time (min)"); plt.ylabel("Bioimpedance (mOhms)")
plt.plot(df_BioZ["time"], df_low_pass_filtered_BioZ3, label="BioZ3-Filtered", color="blue")
plt.plot(df_BioZ["time"], df_BioZ["BioZ3"], label="BioZ3-Raw", color="black", alpha=0.5)
plt.legend(loc="upper right")
plt.grid(True)

plt.subplot(224)
plt.xlabel("Time (min)"); plt.ylabel("Bioimpedance (mOhms)")
plt.plot(df_BioZ["time"], df_low_pass_filtered_BioZ4, label="BioZ4-Filtered", color="orange")
plt.plot(df_BioZ["time"], df_BioZ["BioZ4"], label="BioZ4-Raw", color="black", alpha=0.5)
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# Envelope BioZ Signal and Plot
plt.plot(df_BioZ["time"], df_BioZ["BioZ1"], color='firebrick', label='Raw Signal')
plt.xlabel('Time (min)')
plt.ylabel('Bioimpedance (mohms)')
plt.title("Raw and Enveloped BioZ1 Signal")
plt.plot(df_BioZ["time"], envelope_signal(df_low_pass_filtered_BioZ1), color='navy', lw=3, label='Enveloped Signal')
plt.legend(loc="upper right")
plt.show()

# Envelope PPG Signal and Plot
plt.plot(df_PPG["time"], df_PPG["PPG"],color='firebrick',label='Raw Signal')
plt.xlabel('Time (min)')
plt.ylabel('Voltage (mV)')
plt.title("Raw and Enveloped PPG Signal")
plt.plot(df_PPG["time"], envelope_signal(df_PPG_detrend), color='navy', lw=3,label='Enveloped Signal')
plt.legend(loc="upper right")
plt.show()

######################################## Data Trial01 Finapres BP ########################################
# Plot the FinapresBP vs time
plt.figure()
plt.xlabel("Time (min)"); plt.ylabel("Finapres BP (mmHg)")
plt.plot(df_finapresBP["time"], df_finapresBP["FinapresBP"], label="FinapresBP")
plt.grid(True)
plt.legend()
plt.show()
######################################## Data Trial01 Finapres PPG ########################################
# Plot FinapresPPG vs time
plt.figure()
plt.xlabel("Time (min)"); plt.ylabel("Finapres PPG (a.u.)")
plt.title("Finapres PPG vs Time")
plt.plot(df_finapresPPG["time"], df_finapresPPG["FinapresPPG"], label="FinapresPPG")
plt.legend()
plt.grid(True)
plt.show()

######################################## Data Trial01 PPG ########################################
# Plot PPG vs time

plt.figure()
plt.xlabel("Time (min)"); plt.ylabel("PPG (a.u.)")
plt.title("PPG and Filtered PPG vs Time")
plt.plot(df_PPG["time"], df_PPG["PPG"], label="PPG", color="black")
plt.plot(df_PPG["time"], band_pass_filtered_PPG, label="Filtered-PPG", color="blue")
plt.legend()
plt.grid(True)
plt.show()
