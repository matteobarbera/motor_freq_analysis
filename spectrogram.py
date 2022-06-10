import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, windows

from freq_to_thrust import prettify_plot


def view_file(fname: str, t_window: list = None):
    """
    Visualize to make sure everything works and find offset
    """
    fs, data = wavfile.read(fname)
    start_t, end_t = [0, 0]
    if t_window is not None:
        start_t, end_t = t_window
    else:
        end_t = data.shape[0] / fs
    start_idx = int(start_t * fs)
    end_idx = int(end_t * fs)
    time = np.linspace(0., end_t, len(data[start_idx:end_idx]))
    plt.title(fname)
    plt.plot(time, data[start_idx:end_idx, 0])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


@prettify_plot()
def plot_spectrogram(fname, step_dur, t_window, delta_f=None, calibration=False):
    fs, test_data = get_relevant_data(fname, t_window)
    length = test_data.shape[0] / fs
    nsteps = int(length / step_dur)

    max_subplots = 8
    nfigs = int(nsteps / max_subplots) + 1
    subplots = [plt.subplots(max_subplots // 2, 2) for i in range(nfigs)]
    for i, subplot in enumerate(subplots):
        subplot[0].canvas.set_window_title(fname + f" {i}")
        for ax in subplot[1].reshape(-1):
            # ax.set_yticks([250])
            ax.set_yticks([0, 200, 400, 600])
            ax.set_ylabel("Frequency [Hz]")
            ax.set_xlabel("Time [s]")
        #     ax.grid()
    current_fig = 0
    axs = subplots[current_fig][1]
    for i, (f, t_seg, sxx) in enumerate(compute_spectrogram(test_data, fs, nsteps, step_dur)):
        if i % max_subplots == 0 and i != 0:
            subplots[0][0].tight_layout()
            subplots[0][0].subplots_adjust(left=0.229, wspace=0.233, bottom=0.088, top=0.94, hspace=0.83)
            plt.show()
            quit()
            current_fig += 1
            axs = subplots[current_fig][1]
        r = i - current_fig * max_subplots
        c = 0
        if int(i / (max_subplots // 2)) % 2 != 0:
            r -= max_subplots // 2
            c += 1
        f_slice = np.where(f <= 600)
        if calibration:
            axs[r, c].contourf(t_seg, f[f_slice], sxx[f_slice, :][0], 40)
            axs[r, c].set_ylabel(f"{int(i * (9600 / 20))} ppz", rotation=0, labelpad=30)
        else:
            axs[r, c].contourf(t_seg, f[f_slice], 10 * np.log10(sxx[f_slice, :][0]), 40,
                               cmap='twilight')
            # axs[r, c].set_ylabel(f"{round(i * delta_f, 2)} Hz", rotation=0, labelpad=25)
            axs[r, c].title.set_text(my_bold(f"Signal frequency: {round(i * delta_f, 2)} Hz"))
    plt.show()


def my_bold(text):
    string = str(text).split(" ")
    bold_string = " ".join([r"$\bf{" + word + "}$" for word in string])
    return bold_string


def compute_spectrogram(test_data, fs, nsteps, lstep):
    nwind = 3000
    noverlap = nwind - 600
    for i in range(nsteps + 1):
        s_idx = i * fs * lstep
        e_idx = (i + 1) * fs * lstep
        f, t_seg, sxx = spectrogram(test_data[s_idx:e_idx], fs, window=windows.hann(nwind), noverlap=noverlap)
        yield f, t_seg, sxx


def get_relevant_data(fname, t_window=None):
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
    # # wav_files = [file for file in glob.glob("./wavfiles/*.wav")]
    # wav_files = {"./wavfiles/Calibration_forward.wav": ([10.2, 105.1], 5, None),
    #              "./wavfiles/Calibration_forward_2.wav": ([11.9, 111.7], 5, None),
    #              "./wavfiles/Test_forward.wav": ([11.5, 90.5], 3, 0.5),
    #              "./wavfiles/Calibration_reverse.wav": ([19.6, 104.5], 5, None),
    #              "./wavfiles/Test_reverse.wav": ([10, 72.2], 3, 0.5),
    #              "./wavfiles/Test_forward_2.wav": ([4.6, 394.2], 5, 0.1),
    #              "./wavfiles/Calibration_reverse_2.wav": ([23.2, 113.1], 5, None),
    #              "./wavfiles/Test_reverse_2.wav": ([2.5, 291.7], 5, 0.1)}
    # for fname, (t_wind, step_dur, delta_f) in wav_files.items():
    #     # if "Calibration" in fname:
    #     #     plot_spectrogram(fname, step_dur, t_wind, calibration=True)
    #     if "Calibration" not in fname and "2" in fname:
    #         plot_spectrogram(fname, step_dur, t_wind, delta_f=delta_f)

    # wav_files = [file for file in glob.glob("./wavfiles/19-3/*.wav")]
    wav_files = {"./wavfiles/19-3/SINE_F_50-100.wav":  ([9.63, 171.62], 2, 0.25),
                 "./wavfiles/19-3/BLOCK_F_10-100.wav": ([17.38, 179.37], 2, 0.25),
                 "./wavfiles/19-3/SINE_R_50-100.wav":  ([9.96, 171.95], 2, 0.25),
                 "./wavfiles/19-3/SINE_F_10-100.wav":  ([10.96, 172.95], 2, 0.25),
                 "./wavfiles/19-3/CALIB_F_19-3.wav":   ([13.17, 55.16], 2, 0.25),
                 "./wavfiles/19-3/BLOCK_R_50-100.wav": ([10.97, 172.96], 2, 0.25),
                 "./wavfiles/19-3/BLOCK_F_50-100.wav": ([9.49, 171.18], 2, 0.25),
                 "./wavfiles/19-3/BLOCK_R_10-50.wav":  ([6.82, 168.81], 2, 0.25),
                 "./wavfiles/19-3/BLOCK_R_10-100.wav": ([11.9, 173.89], 2, 0.25),
                 "./wavfiles/19-3/SINE_R_10-100.wav":  ([9.6, 171.59], 2, 0.25),
                 "./wavfiles/19-3/CALIB_R_19-3.wav":   ([16.7, 58.69], 2, 0.25),
                 "./wavfiles/19-3/SINE_R_10-50.wav":   ([10.6, 172.59], 2, 0.25),
                 "./wavfiles/19-3/SINE_F_10-50.wav":   ([7.25, 169.24], 2, 0.25),
                 "./wavfiles/19-3/BLOCK_F_10-50.wav":  ([7.05, 169.04], 2, 0.25)}

    # for f, (t_w, _, _) in wav_files.items():
    #     view_file(f, t_w)
    for fname, (t_wind, step_dur, delta_f) in wav_files.items():
        # if "CALIB" in fname:
        #     plot_spectrogram(fname, step_dur, t_wind, calibration=True)
        if "BLOCK" in fname and "F" in fname:
            plot_spectrogram(fname, step_dur, t_wind, delta_f=delta_f)
        # if "BLOCK" in fname and "R" in fname:
        #     plot_spectrogram(fname, step_dur, t_wind, delta_f=delta_f)
        # if "SINE" in fname and "F" in fname:
        #     plot_spectrogram(fname, step_dur, t_wind, delta_f=delta_f)
        # if "SINE" in fname and "R" in fname:
        #     plot_spectrogram(fname, step_dur, t_wind, delta_f=delta_f)
