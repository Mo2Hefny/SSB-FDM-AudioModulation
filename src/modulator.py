import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write, read
import os
from common_includes import *
from plot_signal import Plotter
from filter_signal import Filterer


carrier_frequencies = [6000, 11000, 16000]  # Carrier frequencies for FDM


if __name__ == "__main__":
    # path filtered signals
    filtered_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "filtered")
    # path modulated signals
    modulated_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "modulated")
    modulated_magnitude_spectrum_dir = os.path.join(modulated_dir, "magnitude_spectrum")
    fdm_magnitude_spectrum_dir = os.path.join(modulated_dir, "fdm_spectrum")
    dsb_magnitude_spectrum_dir = os.path.join(modulated_dir, "dsb_spectrum")


    if not os.path.exists(modulated_dir):
        os.makedirs(modulated_dir)
    if not os.path.exists(modulated_magnitude_spectrum_dir):
        os.makedirs(modulated_magnitude_spectrum_dir)
    if not os.path.exists(fdm_magnitude_spectrum_dir):
        os.makedirs(fdm_magnitude_spectrum_dir)

    # Modulation and FDM
    modulated_signals = []
    for i, file in enumerate(os.listdir(filtered_dir)):
        if file.endswith(".wav"):
            sample_rate, signal = read(os.path.join(filtered_dir, file))
            left_channel = signal[:, 0]  
            #right_channel = signal[:, 1]  
            # Generate carrier signal
            
            t = np.linspace(0, DURATION, sample_rate * DURATION, endpoint=False)  # Time vector
            carrier = np.cos(2 * np.pi * carrier_frequencies[i] * t)


            modulated_left = left_channel * carrier
            #modulated_right = right_channel * carrier


            # Combine the modulated channels back into a stereo signal
            #modulated_stereo_signal = np.column_stack((modulated_left, modulated_right))
            Plotter.plot_magnitude_spectrum(modulated_left, sample_rate, mono=True, title=f"dsb {file} Magnitude Spectrum", save_dir=dsb_magnitude_spectrum_dir, file_name=f"dsb_{file}_magnitude_spectrum.png")
            # Suppress lower sideband
            ssb_signal = Filterer.suppress_lower_sideband(modulated_left, sample_rate, carrier_frequencies[i])
            # plot the magnitude spectrum of the filtered signal
            Plotter.plot_magnitude_spectrum(ssb_signal, sample_rate, mono=True, title=f"Modulated {file} Magnitude Spectrum", save_dir=modulated_magnitude_spectrum_dir, file_name=f"modulated_{file}_magnitude_spectrum.png")
            modulated_signals.append(ssb_signal)

        # Sum all modulated signals to form the FDM signal
    fdm_signal = np.sum(modulated_signals, axis=0)

    Plotter.plot_magnitude_spectrum(fdm_signal, SAMPLE_RATE, mono=True, title=f"FDM {file} Magnitude Spectrum", save_dir=fdm_magnitude_spectrum_dir, file_name=f"FDM_{file}_magnitude_spectrum.png")
    
    # Save the FDM signal
    write(os.path.join(modulated_dir, "fdm_signal.wav"), sample_rate, fdm_signal.astype(np.int16))