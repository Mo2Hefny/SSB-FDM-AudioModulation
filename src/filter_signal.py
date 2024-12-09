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
    def suppress_lower_sideband(cls, signal: np.ndarray, sample_rate: int, carrier_freq: float) -> np.ndarray:
        """
        Suppress all frequencies below the carrier frequency using an ideal high-pass filter.
        This converts the signal to Single Sideband (SSB) by removing the lower sideband.
        
        Parameters:
        - signal: Input signal (1D numpy array).
        - sample_rate: Sampling frequency (Hz).
        - carrier_freq: Carrier frequency around which modulation happens (Hz).
        
        Returns:
        - Filtered signal with the lower sideband suppressed (SSB).
        """
        # Perform Fourier Transform to convert signal to frequency domain
        spectrum = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)

        # Create an ideal high-pass filter (1 for frequencies above the carrier, 0 for below)
        high_pass_filter = np.abs(freqs) >= carrier_freq

        # Apply the high-pass filter (remove lower sideband)
        spectrum[~high_pass_filter] = 0  # Set the lower sideband components to 0

        # Perform Inverse Fourier Transform to get the filtered signal in time domain
        filtered_signal = np.fft.ifft(spectrum).real
        
        return filtered_signal

    @classmethod
    def band_pass_filter(cls, signal: np.ndarray, sample_rate: int, low_cutoff: float, high_cutoff: float) -> np.ndarray:
        """
        Apply an ideal band-pass filter to a signal, passing only the frequencies between low_cutoff and high_cutoff.
        
        Parameters:
        - signal: Input signal (1D numpy array).
        - sample_rate: Sampling frequency (Hz).
        - low_cutoff: Lower bound of the passband (Hz).
        - high_cutoff: Upper bound of the passband (Hz).
        
        Returns:
        - Filtered signal with only frequencies between low_cutoff and high_cutoff passed.
        """
        # Perform Fourier Transform (FFT) of the signal
        spectrum = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)

        # Create an ideal band-pass filter: 1 for frequencies in range, 0 otherwise
        band_pass_filter = (np.abs(freqs) >= low_cutoff) & (np.abs(freqs) <= high_cutoff)

        # Apply the band-pass filter by zeroing out frequencies outside the passband
        spectrum[~band_pass_filter] = 0

        # Perform Inverse Fourier Transform (IFFT) to get the filtered signal in the time domain
        filtered_signal = np.fft.ifft(spectrum).real
        
        return filtered_signal

    @classmethod
    def _low_pass_butterworth(cls, signal: np.ndarray, sample_rate: int, cutoff_frequency: float, order: int=8) -> np.ndarray:
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
    input_magnitude_spectrum_dir = os.path.join(input_dir, "magnitude_spectrum")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "filtered")
    save_magnitude_spectrum_dir = os.path.join(save_dir, "magnitude_spectrum")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # loop through all the files in the data directory
    for file in os.listdir(input_dir):
        if file.endswith(".wav"):
            # read the file
            sample_rate, signal = read(os.path.join(input_dir, file))
            # filter the signal
            filtered_signal = Filterer.low_pass_filter(signal, sample_rate, LIMIT_FREQUENCY, FilterType.LOW_PASS_BUTTERWORTH)
            # save the filtered signal
            write(os.path.join(save_dir, file), sample_rate, filtered_signal.astype(np.int16))
            # plot the magnitude spectrum of the original signal
            Plotter.plot_magnitude_spectrum(signal, sample_rate, title=f"original {file} Magnitude Spectrum", save_dir=input_magnitude_spectrum_dir, file_name=f"original_{file}_magnitude_spectrum.png")
            # plot the magnitude spectrum of the filtered signal
            Plotter.plot_magnitude_spectrum(filtered_signal, sample_rate, title=f"Filtered {file} Magnitude Spectrum", save_dir=save_magnitude_spectrum_dir, file_name=f"filtered_{file}_magnitude_spectrum.png")