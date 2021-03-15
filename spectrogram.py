import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, windows


def view_file(fname: str, t_window: list = None):
    """
    Visualize to make sure everything works and find offset
    """
    fs, data = wavfile.read(fname)
    start_t, end_t = [0, 0]
    if t_window is not None:
        start_t, end_t = t_window
    start_idx = int(start_t * fs)
    end_idx = int(end_t * fs)
    time = np.linspace(0., end_t, len(data[start_idx:end_idx]))
    plt.plot(time, data[start_idx:end_idx, 0])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


def plot_spectrogram(fname, step_dur, t_window, calibration=False):
    fs, test_data = get_relevant_data(fname, t_window)
    length = test_data.shape[0] / fs
    nsteps = int(length / step_dur)

    max_subplots = 8
    nfigs = int(nsteps / max_subplots) + 1
    subplot_rows = nsteps // 2
    if nsteps % 2 != 0:
        subplot_rows += 1
    subplots = [plt.subplots(max_subplots // 2, 2) for i in range(nfigs)]
    for i, subplot in enumerate(subplots):
        subplot[0].canvas.set_window_title(fname + f" {i}")
        for ax in subplot[1].reshape(-1):
            ax.set_yticks([250])
        #     ax.grid()
    current_fig = 0
    axs = subplots[current_fig][1]
    for i, (f, t_seg, sxx) in enumerate(compute_spectrogram(test_data, fs, nsteps, step_dur)):
        if i % max_subplots == 0 and i != 0:
            current_fig += 1
            axs = subplots[current_fig][1]
        r = i - current_fig * max_subplots
        c = 0
        if int(i / (max_subplots // 2)) % 2 != 0:
            r -= max_subplots // 2
            c += 1
        f_slice = np.where(f <= 600)
        if calibration:
            axs[r, c].contourf(t_seg, f[f_slice], sxx[f_slice, :][0], 40, shading='auto')
            axs[r, c].set_ylabel(f"{int(i * (9600 / 20))} ppz", rotation=0, labelpad=30)
        else:
            axs[r, c].contourf(t_seg, f[f_slice], 10 * np.log10(sxx[f_slice, :][0]), 40, shading='auto',
                               cmap='twilight')
            axs[r, c].set_ylabel(f"{0.5 + i * 0.5} Hz", rotation=0, labelpad=25)
    plt.show()


def compute_spectrogram(test_data, fs, nsteps, lstep):
    nwind = 3000
    noverlap = nwind - 600
    for i in range(nsteps + 1):
        s_idx = i * fs * lstep
        e_idx = (i + 1) * fs * lstep
        f, t_seg, sxx = spectrogram(test_data[s_idx:e_idx], fs, window=windows.hann(nwind), noverlap=noverlap)
        yield f, t_seg, sxx


def get_relevant_data(fname, t_window):
    fs, data = wavfile.read(fname)
    start_t, end_t = [0, 0]
    if t_window is not None:
        start_t, end_t = t_window
    start_idx = int(start_t * fs)
    end_idx = int(end_t * fs)

    test_data = data[start_idx:end_idx, 0]
    return fs, test_data


if __name__ == "__main__":
    mpl.rcParams["agg.path.chunksize"] = 10000
    # wav_files = [file for file in glob.glob("./wavfiles/*.wav")]
    wav_files = {"./wavfiles/Calibration_forward.wav": [10.2, 105.1],
                 "./wavfiles/Calibration_forward_2.wav": [11.9, 111.7],
                 "./wavfiles/Test_forward.wav": [11.5, 90.5],
                 "./wavfiles/Calibration_reverse.wav": [19.6, 104.5],
                 "./wavfiles/Test_reverse.wav": [10, 72.2],
                 "./wavfiles/Test_forward_2.wav": [3.1, 394.2],
                 "./wavfiles/Calibration_reverse_2.wav": [23.2, 113.1],
                 "./wavfiles/Test_reverse_2.wav": [2.5, 291.7]}
    for fname, t_wind in wav_files.items():
        if "Calibration" in fname:
            plot_spectrogram(fname, 5, t_wind, calibration=True)
        # if "Calibration" not in fname:
        #     plot_spectrogram(fname, 3, t_wind)
