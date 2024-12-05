import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from common_includes import *

class Plotter:
    @classmethod
    def plot(cls, time, signal, title=None, x_label=None, y_label=None):
        plt.title(title)
        plt.plot(time, signal)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    @classmethod
    def plot_wavfile(cls, filename, title=None, x_label=None, y_label=None):
        sample_rate, signal = wavfile.read(filename)
        if len(signal.shape) == 2:
            signal = signal[:, 0]
        time = np.arange(len(signal)) / sample_rate
        cls.plot(time, signal, title, x_label, y_label)

    @classmethod
    def plot_wavfile_magnitude_spectrum(cls, filename, title=None, x_label=None, y_label=None):
        sample_rate, signal = wavfile.read(filename)
        if len(signal.shape) == 2:
            signal = signal[:, 0]
        cls.plot_magnitude_spectrum(signal, sample_rate, title, x_label, y_label)
            
    @classmethod
    def plot_magnitude_spectrum(cls, signal, sample_rate=SAMPLE_RATE, title=None, x_label=None, y_label=None):
        fft_signal = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal), 1 / sample_rate)
        magnitude = np.abs(fft_signal)

        plt.figure(figsize=(10, 6))
        plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid()
        plt.show()
