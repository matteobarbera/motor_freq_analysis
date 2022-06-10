import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import glob
from scipy.signal import spectrogram, windows
from scipy.signal import welch, find_peaks, peak_prominences

from spectrogram import *


if __name__ == "__main__":
    mpl.rcParams["agg.path.chunksize"] = 10000

    # wav_files = [file for file in glob.glob("./wavfiles/31-5/*.wav")]
    # for f in wav_files:
    #     view_file(f)
    # quit()
    wav_files = {"./wavfiles/31-5/SINE_F_50-100.wav": ([5.59, 188.58], 3, 0.25),
                 "./wavfiles/31-5/BLOCK_F_10-100.wav": ([9.05, 192.04], 3, 0.25),
                 "./wavfiles/31-5/SINE_R_50-100.wav": ([15.27, 198.26], 3, 0.25),
                 "./wavfiles/31-5/SINE_F_10-100.wav": ([11.32, 184.31], 3, 0.25),
                 "./wavfiles/31-5/Ale_5-100.wav": ([2.07, 8.73], 3, 0.25),
                 "./wavfiles/31-5/CALIB_R_31-5.wav": ([14.2, 56.19], 2, 0.25),
                 "./wavfiles/31-5/BLOCK_R_50-100.wav": ([12.35, 185.34], 3, 0.25),
                 "./wavfiles/31-5/BLOCK_F_50-100.wav": ([5.06, 188.05], 3, 0.25),
                 "./wavfiles/31-5/BLOCK_R_10-50.wav": ([7.27, 190.26], 3, 0.25),
                 "./wavfiles/31-5/BLOCK_R_10-100.wav": ([11.89, 184.88], 3, 0.25),
                 "./wavfiles/31-5/SINE_R_10-100.wav": ([6.33, 189.32], 3, 0.25),
                 "./wavfiles/31-5/CALIB_F_31-5.wav": ([6.3, 48.29], 2, 0.25),
                 "./wavfiles/31-5/SINE_R_10-50.wav": ([12.65, 195.64], 3, 0.25),
                 "./wavfiles/31-5/SINE_F_10-50.wav": ([6.67, 189.66], 3, 0.25),
                 "./wavfiles/31-5/Ale_50-100.wav": ([2.4, 9.29], 3, 0.25),
                 "./wavfiles/31-5/BLOCK_F_10-50.wav": ([11.23, 184.22], 3, 0.25)}

    for fname, (t_wind, step_dur, delta_f) in wav_files.items():
        plt.rcParams.update({"font.size": 13,
                             "axes.titlesize": 12})
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
        # if "Ale" in fname:
        #     pass
        #     # print(fname)
        #     # plot_ale(fname, t_wind)
