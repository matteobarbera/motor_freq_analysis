from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import blackman
from scipy.signal import find_peaks
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np


def view_file(fname: str):
    """
    Visualize to make sure everything works and find offset
    """
    fs, data = wavfile.read(fname)
    length = data.shape[0] / fs
    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data[:, 0])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


def calibration_analysis(fname: str, t_window: list, step_duration: float, n_steps: int):
    fs, data = wavfile.read(fname)
    # tot_time = data.shape[0] / fs
    step_length = step_duration * fs  # step duration comes from C file
    start_idx = int(t_window[0] * fs)
    end_idx = int(t_window[1] * fs)
    track1 = data[start_idx:end_idx, 0]

    peaks = []
    plt.figure()
    # Compute FFTs for each test step
    for i in range(n_steps - 1):
        subtrack = track1[i * step_length:(i + 1) * step_length]
        n = len(subtrack)
        wind = blackman(n)
        yf = rfft(subtrack)
        xf = rfftfreq(n, fs)
        plt.semilogy(xf, 2.0 / n * np.abs(yf), label=f'{i} step')

        # Log frequency and amplitude of each peak
        peaks_idx = find_peaks(yf, height=0)
        peaks.append([xf[peaks_idx[0]], yf[peaks_idx[0]]])
        plt.show()
    # Plot frequencies of peaks for each test step
    # They should stay constant between steps
    # For actual test isolate square pulses (min/max thrust value)?
    plt.figure()
    for i, p in enumerate(peaks):
        plt.plot(np.asarray(np.repeat(i, len(p[0])), dtype=int), p[0], ls="None", marker="x", markevery=1)
    plt.ylabel("f [Hz]")
    plt.xlabel("Test step")
    plt.show()


if __name__ == "__main__":
    # For calibration
    view_file("calib")
    calibration_analysis("calib", 0, 5, 20)

    # For actual test
    view_file("test")
    calibration_analysis("test", 0, 3, int(15 // 0.5))
