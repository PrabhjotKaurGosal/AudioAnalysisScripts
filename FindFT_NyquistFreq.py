# This code finds the Fourier Tranform of a signal and the Nyquist frequency

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa as lr
from scipy import signal
from scipy.fft import fft, ifft
import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
        #  Read the audio files
    ch1, sfreq = lr.load("ch1.wav", sr=44100)
    ch2, sfreq = lr.load("ch2.wav", sr=44100)
    ch3, sfreq = lr.load("ch3.wav", sr=44100)
  
    # # # Find the spectrogram of the signal 
    f,t, Sxx  = signal.spectrogram(ch1, fs=sfreq)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    ind_max = np.unravel_index(np.argmax(Sxx, axis=None), Sxx.shape) 
    ind_min = np.unravel_index(np.argmin(Sxx, axis=None), Sxx.shape) 
    row_max = ind_max[0]
    col_max = ind_max[1]
    row_min = ind_min[0]
    col_min = ind_min[1]
  
    Bandwidth = Sxx[row_max][col_max] - Sxx[row_min][col_min]
    fsample = 2*Bandwidth  # This is called Nyquist frequency
    print("The sampling frequency of the signal must be greater than: ", fsample)