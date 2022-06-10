from scipy.io import wavfile
from scipy.signal import spectrogram, windows

from spectrogram import view_file


def compute_spectrogram(test_data, fs):
    nwind = 3000
    noverlap = nwind - 600
    f, t_seg, sxx = spectrogram(test_data, fs, window=windows.hann(nwind), noverlap=noverlap)
    return f, t_seg, sxx


