import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write, read
import os
from common_includes import *
from plot_signal import Plotter


class Filterer:
    @classmethod
    def low_pass_filter(cls, signal: np.ndarray, sample_rate: int, cutoff_frequency: float, mode: FilterType=FilterType.LOW_PASS_BUTTERWORTH) -> np.ndarray:
        """
        Apply a low-pass filter to a signal.

        Args:
            signal (np.ndarray): The signal to filter.
            cutoff_frequency (float): The cutoff frequency of the filter.
            mode (FilterType): The type of filter to apply.

        Returns:
            np.ndarray: The filtered signal.
        """
        if mode == FilterType.LOW_PASS_FOURIER:
            return cls._low_pass_fourier(signal, sample_rate, cutoff_frequency)
        elif mode == FilterType.LOW_PASS_BUTTERWORTH:
            return cls._low_pass_butterworth(signal, sample_rate, cutoff_frequency)
        else:
            raise ValueError(f"Invalid filter mode: {mode}")
    
    @classmethod
    def _low_pass_fourier(cls, signal: np.ndarray, sample_rate: int, cutoff_frequency: float) -> np.ndarray:
        """
        Apply a low-pass Fourier filter to a signal.

        Args:
            signal (np.ndarray): The signal to filter.
            cutoff_frequency (float): The cutoff frequency of the filter.

        Returns:
            np.ndarray: The filtered signal.
        """
        print("Applying Fourier filter")
        nyquist = 0.5 * sample_rate
        cutoff_idx = int(cutoff_frequency / nyquist * (len(signal) // 2))
        signal_fft = np.abs(np.fft.rfft(signal, axis=0))
        freqs = np.fft.rfftfreq(len(signal), d=1/sample_rate)
        signal_fft[cutoff_idx:] = 0
        filtered_signal = np.fft.irfft(signal_fft, n=len(signal), axis=0)
        return filtered_signal

    @classmethod
    def _low_pass_butterworth(cls, signal: np.ndarray, sample_rate: int, cutoff_frequency: float, order: int=6) -> np.ndarray:
        """
        Apply a low-pass Butterworth filter to a signal.
        
        Args:
            signal (np.ndarray): The signal to filter.
            cutoff_frequency (float): The cutoff frequency of the filter.
        
        Returns:
            np.ndarray: The filtered signal.
        """
        if len(signal.shape) > 1:
            signal_left = signal[:, 0]
            signal_right = signal[:, 1]
        else:
            signal_left = signal
            signal_right = None
        
        # Design the low-pass filter
        nyquist_rate = sample_rate / 2.0  # Nyquist frequency
        normal_cutoff = cutoff_frequency / nyquist_rate  # Normalize cutoff frequency

        # Validate cutoff frequency
        if not (0 < normal_cutoff < 1):
            raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency.")

        # Design Butterworth filter
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # Apply the filter to the signal
        filtered_left = filtfilt(b, a, signal_left)
        filtered_right = None
        if signal_right is not None:
            filtered_right = filtfilt(b, a, signal_right)
        
        # Combine channels back (stereo if both channels exist)
        if filtered_right is not None:
            filtered_signal = np.vstack((filtered_left, filtered_right)).T
        else:
            filtered_signal = filtered_left
        
        return filtered_signal


if __name__ == "__main__":
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "input")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "filtered")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # loop through all the files in the data directory
    for file in os.listdir(input_dir):
        if file.endswith(".wav"):
            # read the file
            sample_rate, signal = read(os.path.join(input_dir, file))
            # filter the signal
            filtered_signal = Filterer.low_pass_filter(signal, sample_rate, 4000, FilterType.LOW_PASS_BUTTERWORTH)
            # save the filtered signal
            write(os.path.join(save_dir, file), sample_rate, filtered_signal.astype(np.int16))
            # plot the filtered signal
            # Plotter.plot_signal(filtered_signal, sample_rate, title=f"Filtered {file}")
            # plot the magnitude spectrum of the filtered signal
            Plotter.plot_magnitude_spectrum(filtered_signal, sample_rate, title=f"Filtered {file} Magnitude Spectrum")