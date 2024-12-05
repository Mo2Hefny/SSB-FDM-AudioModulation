import numpy as np
from common_includes import *
from plot_signal import Plotter


class Filterer:
    @classmethod
    def low_pass_filter(cls, signal: np.ndarray, cutoff_frequency: float, mode: FilterType=FilterType.LOW_PASS_FOURIER) -> np.ndarray:
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
            return cls._low_pass_fourier(signal, cutoff_frequency)
        elif mode == FilterType.LOW_PASS_BUTTERWORTH:
            return cls._low_pass_butterworth(signal, cutoff_frequency)
        else:
            raise ValueError(f"Invalid filter mode: {mode}")
    
    @classmethod
    def _low_pass_fourier(cls, signal: np.ndarray, cutoff_frequency: float) -> np.ndarray:
        pass