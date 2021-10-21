# This code calculates the time delay between two signals received by microphones spaced apart from each other,
# using various Cross Correlation techniques
# Method 1: Simple CRoss Correlation
# Method 2: Generalized Cross Correlation -PHAT (GCC-PHAT)
# Method 3: Generalized Cross Correlation -Maximum Likelihood with Coherence (GCC-MMLC)

import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import wave
import librosa as lr
from scipy import signal
import math
    
# Computer the time delay between two signals using the Generalized Cross Correlation method: GCC-PHAT
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)  # Fourier Transform of sig
    REFSIG = np.fft.rfft(refsig, n=n) # Fourier Transform of refsig
    R = SIG * np.conj(REFSIG) # Cross Correlation in frequency domain (This is also called the Cross Spectral Density)

    # The formula for GCC-PHAT in time domain is = INVERSE_FOURIER ([(SIG)*conj(REFSIG)]/ [magnitude(SIG)*magnitude(conj(REFSIG))]) 
    # That is, GCC-PHAT in tme domain = INVERSE_FOURIER(R/magnitide(R))
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n)) # GCC-PHAT in time domain 
   
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc
    
def CalculateTimeDifferenceOfArrival(ch1,ch2,ch3, sfreq):
    ###### Calculate time delay between channel 1 and channel 2
    y1 = ch1
    y2 = ch2

    # Method 1: Simple Cross Correlation
    correlation = signal.correlate(y2, y1, mode="full")
    lags = signal.correlation_lags(y1.size, y2.size, mode="full")
    lag = lags[np.argmax(correlation)]

    channelNames = ["ch1", "ch2"]
    if lag<0:
        print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
    elif lag>0:
        print("The signal ", channelNames[0], " leads signal ",channelNames[1])
    else:
        print("The lag is Zero between the two signals")
    print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " using simple CC is: ", lag/sfreq, "seconds")
    T_12_cc = lag/sfreq

    # Method 2: Generalized Cross Correlation -PHAT (GCC-PHAT)
    tau, cc = gcc_phat(y2,y1,sfreq)
    if tau<0:
        print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
    elif tau>0:
        print("The signal ", channelNames[0], " leads signal ",channelNames[1])
    else:
        print("The lag is Zero between the two signals")
    print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " using GCC-PHAT is: ", tau, "seconds")
    T_12_gccPhat = tau

    ###### Calcualte time delay between channel 1 and channel 3
    y1 = ch1
    y2 = ch3

    # Method 1: Simple Cross Correlation
    correlation = signal.correlate(y2, y1, mode="full")
    lags = signal.correlation_lags(y1.size, y2.size, mode="full")
    lag = lags[np.argmax(correlation)]

    channelNames = ["ch1", "ch3"]
    if lag<0:
        print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
    elif lag>0:
        print("The signal ", channelNames[0], " leads signal ",channelNames[1])
    else:
        print("The lag is Zero between the two signals")
    print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " is: ", lag/sfreq, "seconds")
    T_13_cc = lag/sfreq

    # Method 2: Generalized Cross Correlation -PHAT (GCC-PHAT)
    tau, cc = gcc_phat(y2,y1,sfreq)
    if tau<0:
        print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
    elif tau>0:
        print("The signal ", channelNames[0], " leads signal ",channelNames[1])
    else:
        print("The lag is Zero between the two signals")
    print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " using GCC-PHAT is: ", tau, "seconds")
    T_13_gccPhat = tau

    # Calculate time delay between channel 2 and channel 3
    y1 = ch2
    y2 = ch3

    # Method 1: Simple Cross Correlation
    correlation = signal.correlate(y2, y1, mode="full")
    lags = signal.correlation_lags(y1.size, y2.size, mode="full")
    lag = lags[np.argmax(correlation)]

    channelNames = ["ch2", "ch3"]
    if lag<0:
        print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
    elif lag>0:
        print("The signal ", channelNames[0], " leads signal ",channelNames[1])
    else:
        print("The lag is Zero between the two signals") 
    print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " is: ", lag/sfreq, "seconds")
    T_23_cc = lag/sfreq

    # Method 2: Generalized Cross Correlation -PHAT (GCC-PHAT)
    tau, cc = gcc_phat(y2,y1,sfreq)
    if tau<0:
        print("The signal ", channelNames[0], " lags behind signal ", channelNames[1])
    elif tau>0:
        print("The signal ", channelNames[0], " leads signal ",channelNames[1])
    else:
        print("The lag is Zero between the two signals")
    print("The time delay between the signals ", channelNames[0], " and ", channelNames[1], " using GCC-PHAT is: ", tau, "seconds")
    T_23_gccPhat = tau
    
    return T_13_cc,T_23_cc, T_12_cc,T_13_gccPhat, T_23_gccPhat, T_12_gccPhat    

if __name__ == "__main__":
        #  Read the audio files
    ch1, sfreq = lr.load("ch1.wav")
    ch2, sfreq = lr.load("ch2.wav")
    ch3, sfreq = lr.load("ch3.wav")

    #tau, cc = gcc_phat(ch1,ch2,sfreq)
    T_13,T_23, T_12, T_13_gcc,T_23_gcc, T_12_gcc  = CalculateTimeDifferenceOfArrival(ch1,ch2,ch3, sfreq)
    #print("Results from GCC: ", tau)
    print("Results from CC: ", T_13, T_23, T_12)
    print("Results from GCC are: ", T_13_gcc,T_23_gcc, T_12_gcc )
