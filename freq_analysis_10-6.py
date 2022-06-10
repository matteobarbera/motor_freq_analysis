from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch, find_peaks, spectrogram, windows
from scipy.optimize import curve_fit

from freq_to_thrust import prettify_plot


def get_data(filename):
    fs, data = wavfile.read(filename)
    test_data = data[:, 0]
    return fs, test_data


def compute_spectrogram(test_data, fs):
    nwind = 5000
    noverlap = nwind - 800
    return spectrogram(test_data, fs, window=windows.hann(nwind), noverlap=noverlap)


def filter_spectrogram(f_sxx: np.ndarray, sxx: np.ndarray) -> np.ndarray:
    """ Find highest power frequency per time segment """
    sxx_filt = []
    for f in sxx.T:  # SXX has shape [nFreqs, nTime]
        idx_maxf = np.argmax(f)
        sxx_filt.append(f_sxx[idx_maxf])

    return np.asarray(sxx_filt)


def get_only_test_data(data, fs):
    test_duration = 60

    f, t_seg, sxx = compute_spectrogram(data, fs)

    f_slice = np.where((300 < f) & (f <= 1100))
    sxx_filt = filter_spectrogram(f[f_slice], sxx[f_slice, :][0])

    t_slice = np.where(t_seg < 20)
    max_f = max(sxx_filt[t_slice])
    max_f_idx = np.argwhere(sxx_filt == max_f)

    t_start = t_seg[max_f_idx[0, 0]]

    idx_start = int(t_start * fs)
    idx_end = int((t_start + test_duration) * fs)

    return data[idx_start:idx_end]


def plot_pxx(f_pxx: np.ndarray, pxx: np.ndarray):
    plt.figure()

    ax = plt.gca()

    ax.semilogx(f_pxx, pxx, color="C0")
    ax.set_title("PXX")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PXX [V^2 / Hz]")
    return ax


def first_order_objective(x, a, b, c):
    return a * x + b


def second_order_objective(x, a, b, c):
    return a * x ** 2 + b * x + c


def third_order_objective(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


@prettify_plot()
def foo(filename):
    fs, test_data = get_data(filename)
    test_data = get_only_test_data(test_data, fs)

    f, t_seg, sxx = compute_spectrogram(test_data, fs)

    f_slice = np.where((300 < f) & (f <= 1100))

    sxx_filt = filter_spectrogram(f[f_slice], sxx[f_slice, :][0])
    peaks_idx, _ = find_peaks(sxx_filt, prominence=290, distance=23)
    troughs_idx, _ = find_peaks(-sxx_filt, prominence=370, distance=23)

    peaks_idx = peaks_idx[:-1]
    troughs_idx = troughs_idx[:-1]

    peaks_popt, _ = curve_fit(third_order_objective, t_seg[peaks_idx], sxx_filt[peaks_idx])
    a, b, c, d = peaks_popt
    peaks_best_fit = third_order_objective(t_seg, a, b, c, d)

    troughs_popt, _ = curve_fit(third_order_objective, t_seg[troughs_idx], sxx_filt[troughs_idx])
    a, b, c, d = troughs_popt
    troughs_best_fit = third_order_objective(t_seg, a, b, c, d)

    plt.figure()
    ax = plt.gca()
    ax.scatter(t_seg, sxx_filt, alpha=0.6, color="silver")
    ax.scatter(t_seg[peaks_idx], sxx_filt[peaks_idx], alpha=0.8, color="indianred")
    ax.scatter(t_seg[troughs_idx], sxx_filt[troughs_idx], alpha=0.8, color="forestgreen")
    ax.plot(t_seg, peaks_best_fit, color='k', lw=3)
    ax.plot(t_seg, troughs_best_fit, color='k', lw=3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

    gain = peaks_best_fit - troughs_best_fit
    gain /= max(gain)
    freq = np.linspace(0.6, 10, num=len(gain))
    plt.figure()
    plt.plot(freq, gain)
    plt.ylabel("Gain")
    plt.xlabel("Frequency [Hz]")
    plt.ylim(bottom=0)

    plt.figure()
    ax = plt.gca()
    ax.contourf(t_seg, f[f_slice], 10 * np.log10(sxx[f_slice, :][0]), 100, cmap='twilight')


if __name__ == "__main__":
    forward = "./wavfiles/10-6/Forward_22-06-10_06-10f_5500_3400_60s.wav"
    reverse = "./wavfiles/10-6/Reverse_22-06-10_06-10f_5500_3400_60s.wav"

    foo(forward, context="talk")
    # foo(reverse)
    plt.show()
