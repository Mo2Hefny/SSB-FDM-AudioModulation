"""
This file contains the common includes for the project.
"""
from enum import Enum

DURATION = 10
SAMPLE_RATE = 44100
LIMIT_FREQUENCY = 4000
TOLERANCE_FREQUENCY = 500

class FilterType(Enum):
    LOW_PASS_FOURIER = 1
    LOW_PASS_BUTTERWORTH = 2
    HIGH_PASS = 3
    BAND_PASS = 4
    BAND_STOP = 5