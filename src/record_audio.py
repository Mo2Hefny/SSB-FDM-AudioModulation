"""
This module records audio from the microphone and saves it to a file.
"""
import sounddevice as sd
from scipy.io.wavfile import write
import sys
import os
from common_includes import *


def record_audio(filename: str) -> None:
    """
    Record audio from the microphone and save it to a file.

    Args:
        filename (str): The path to save the audio file to.
    """
    input("Press Enter to start recording...")
    print(f"Recording audio to {filename} for {DURATION} seconds")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=2, dtype='int16')
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    print(f"Audio saved to {filename}")
    print()


if __name__ == "__main__":
    is_test = "--test" in sys.argv
    save_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "data", "input")
    if is_test:
        from plot_signal import Plotter
        test_filename = os.path.join(save_dir, "test_input.wav")
        record_audio(test_filename)
        Plotter.plot_wavfile(
            test_filename, "Audio Waveform", "Time (Seconds)", "Amplitude")
        Plotter.plot_wavfile_magnitude_spectrum(
            test_filename, "Audio Magnitude Spectrum", "Frequency (Hz)", "Magnitude")
    else:
        record_audio(os.path.join(save_dir, "input1.wav"))
        record_audio(os.path.join(save_dir, "input2.wav"))
        record_audio(os.path.join(save_dir, "input3.wav"))
