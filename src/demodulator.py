import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write, read
import os
from common_includes import *
from plot_signal import Plotter
from filter_signal import Filterer


carrier_frequencies = [6000, 11000, 16000]  # Carrier frequencies for FDM


if __name__ == "__main__":
    # path modulated signals
    modulated_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "modulated")
    # path demodulated signals
    demodulated_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "demodulated")
    separated_spectrums_dir = os.path.join(demodulated_dir, "separated_spectrums")
    separated_audios_dir = os.path.join(demodulated_dir, "separated_audios")
    signals_after_demodulation_spectrum_dir = os.path.join(demodulated_dir, "signals_after_demodulation_spectrum")
    signals_after_demodulation_audios_dir = os.path.join(demodulated_dir, "signals_after_demodulation_audios")

    if not os.path.exists(demodulated_dir):
        os.makedirs(demodulated_dir)

    if not os.path.exists(separated_spectrums_dir):
        os.makedirs(separated_spectrums_dir)
    
    if not os.path.exists(separated_audios_dir):
        os.makedirs(separated_audios_dir)

    if not os.path.exists(signals_after_demodulation_spectrum_dir):
        os.makedirs(signals_after_demodulation_spectrum_dir)

    if not os.path.exists(signals_after_demodulation_audios_dir):
        os.makedirs(signals_after_demodulation_audios_dir)


    for file in os.listdir(modulated_dir):
        if file.endswith(".wav"):
            sample_rate, signal = read(os.path.join(modulated_dir, file))
            # iterate through the carrier frequencies and do band pass filter from filter_signal.py and add them to signalsarray
            for i , carrier_frequency in enumerate(carrier_frequencies):
                band_pass_signal = Filterer.band_pass_filter(signal, sample_rate, carrier_frequency, carrier_frequency + LIMIT_FREQUENCY + TOLERANCE_FREQUENCY)
                # plot the magnitude spectrum of the filtered signal
                Plotter.plot_magnitude_spectrum(band_pass_signal, sample_rate, mono=True, title=f"Band Pass {carrier_frequency} Magnitude Spectrum", save_dir=separated_spectrums_dir, file_name=f"band_pass_{carrier_frequency}_magnitude_spectrum.png")
                # save the filtered signal
                write(os.path.join(separated_audios_dir, f"separated_signal_{i}.wav"), sample_rate, band_pass_signal.astype(np.int16))
    
    for i, file in enumerate(os.listdir(separated_audios_dir)):
        if file.endswith(".wav"):
            sample_rate, signal = read(os.path.join(separated_audios_dir, file))
            
            t = np.linspace(0, DURATION, sample_rate * DURATION, endpoint=False)  # Time vector
            carrier = np.cos(2 * np.pi * carrier_frequencies[i] * t)


            demodulated_signal = signal * carrier

            demodulated_signal = Filterer.low_pass_filter(demodulated_signal, sample_rate, LIMIT_FREQUENCY, FilterType.LOW_PASS_BUTTERWORTH)
            # plot the magnitude spectrum of the filtered signal
            Plotter.plot_magnitude_spectrum(demodulated_signal, sample_rate, mono=True, title=f"Demodulated {file} Magnitude Spectrum", save_dir=signals_after_demodulation_spectrum_dir, file_name=f"out{i}.png")

            write(os.path.join(signals_after_demodulation_audios_dir, f"out{i+1}.wav"), sample_rate, demodulated_signal.astype(np.int16))