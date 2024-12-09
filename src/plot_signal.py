import numpy as np
import os
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
    
        if len(signal.shape) == 1:  # Mono
            cls.plot_magnitude_spectrum(signal, sample_rate, title, x_label, y_label, mono=True)
        elif len(signal.shape) == 2 and signal.shape[1] == 2:  # Stereo
            cls.plot_magnitude_spectrum(signal, sample_rate, title, x_label, y_label, mono=False)
        else:
            raise ValueError("Unsupported audio format")
            
    @classmethod
    def plot_magnitude_spectrum(cls, signal, sample_rate=SAMPLE_RATE, title=None, x_label=None, y_label=None, mono=False, save_dir=None, file_name="magnitude_spectrum.png"):
        # If mono, work with the single channel; if stereo, split the channels
        if mono:
            data_left = signal
            data_right = None
        else:
            data_left = signal[:, 0]
            data_right = signal[:, 1]

        # Calculate frequency bins using FFT
        frequencies = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
        
        # Compute the magnitude spectrum for left and right channels
        magnitude_left = np.abs(np.fft.rfft(data_left))
        magnitude_right = np.abs(np.fft.rfft(data_right)) if data_right is not None else None

        # Plotting the frequency domain
        plt.figure(figsize=(12, 6))
        
        # Plot for left channel
        plt.subplot(1, 2 if not mono else 1, 1)
        plt.plot(frequencies, magnitude_left, label='Left Channel')
        plt.title(f'{title} - Left Channel')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        
        # Plot for right channel if stereo
        if not mono and magnitude_right is not None:
            plt.subplot(1, 2, 2)
            plt.plot(frequencies, magnitude_right, label='Right Channel')
            plt.title(f'{title} - Right Channel')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()

        # Adjust the x-axis range to zoom in on the higher frequencies (e.g., 0-50 kHz)
        #plt.xlim(0, 60000)  # Adjust this range as needed to include carrier frequencies
        
        plt.tight_layout()

         # Save the plot to the specified directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
            save_path = os.path.join(save_dir, file_name)
            plt.savefig(save_path)

        #plt.show()
