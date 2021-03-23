import glob
import re

import numpy as np
import pandas
from matplotlib import pyplot as plt


def process_calibration(fname: str):
    with open(fname) as csv_f:
        data = pandas.read_csv(csv_f, delimiter=',', skiprows=1, na_values=['']).dropna()
    data = data.to_numpy()

    fwd_coeffs = np.polyfit(data[:, 1], data[:, 0], 1)
    rev_coeffs = np.polyfit(data[:, 2], data[:, 0], 1)
    def hz_to_f_fwd(x): return fwd_coeffs[0] * x + fwd_coeffs[1]
    def hz_to_f_rev(x): return rev_coeffs[0] * x + rev_coeffs[1]
    return hz_to_f_fwd, hz_to_f_rev


def pprz_to_thrust(x):
    return 1 / 96 * x


def process_tests():
    hz2f_fwd, hz2f_rev = process_calibration(calibration_csv)
    test_csvs = glob.glob("./test_csv/*.csv")
    freq_dict = {}
    for test_csv in test_csvs:
        fname = re.findall(r'(?<=/)([\w-]*\.csv)', test_csv)[0].split('.')[0]
        data = pandas.read_csv(test_csv, delimiter=',', header=None).to_numpy()
        freq_dict[fname] = data[:, 1:]
    thr_dict = {}
    for name, freq in freq_dict.items():
        if "_f_" in name:
            thr_dict[name] = pprz_to_thrust(hz2f_fwd(freq))
        else:
            thr_dict[name] = pprz_to_thrust(hz2f_rev(freq))
    return freq_dict, thr_dict


def plot_test_label(*args):
    d = {'block': 'Block', 'sine': 'Sine', 'r': 'reverse', 'f': 'forward'}
    return f'{d[args[0]]} {d[args[1]]}'


def plot_tests(data_dict: dict, test_range: str, error_bar: float):
    plt.figure(f'Chirp test for thrust {test_range}%')
    chirp_freq = list(range(11))
    marker_type = ['^', 's', 'v', 'o']
    color = ['k', 'm', 'c', 'r']
    idx = 0
    for tname, tdata in data_dict.items():
        if test_range in tname:
            label = plot_test_label(*tname.split('_'))
            if tdata.shape[1] == 2:
                plt.errorbar(chirp_freq, tdata[:, 0], yerr=error_bar, label=label, color=color[idx], ls='--', marker=marker_type[idx],
                             markevery=1)
                plt.errorbar(chirp_freq, tdata[:, 1], yerr=error_bar, color=color[idx], ls='--', marker=marker_type[idx],
                             markevery=1)
            else:
                plt.errorbar(chirp_freq, tdata, yerr=error_bar, label=label, color=color[idx], ls='--', marker=marker_type[idx], markevery=1)
            idx += 1
    plt.legend()
    plt.title(f'Chirp test for thrust {test_range}%')
    plt.ylabel('Thrust [%]')
    plt.xlabel('Chirp frequency [Hz]')
    plt.grid()


if __name__ == "__main__":
    calibration_csv = "./calibration.csv"
    freqs, thrs = process_tests()
    plot_tests(thrs, '10-100', 4.5)
    plot_tests(thrs, '10-50', 4.5)
    plot_tests(thrs, '50-100', 4.5)
    plt.show()
