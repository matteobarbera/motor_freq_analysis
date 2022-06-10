import glob
import re

import numpy as np
import pandas
from matplotlib import pyplot as plt
import seaborn as sns
from functools import wraps
from itertools import cycle


def prettify_plot():

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if "context" in kwargs.keys():
                _context = kwargs["context"]
                del kwargs["context"]
            else:
                _context = "notebook"

            if "style" in kwargs.keys():
                _style = kwargs["style"]
                del kwargs["style"]
            else:
                _style = "whitegrid"

            if "params" in kwargs.keys():
                _params = kwargs["params"]
                del kwargs["params"]
            else:
                _params = None

            _default_params = {
              # "xtick.bottom": True,
              # "ytick.left": True,
              # "xtick.color": ".8",  # light gray
              # "ytick.color": ".15",  # dark gray
              "axes.spines.left": False,
              "axes.spines.bottom": False,
              "axes.spines.right": False,
              "axes.spines.top": False,
              }
            if _params is not None:
                merged_params = {**_params, **_default_params}
            else:
                merged_params = _default_params
            with sns.plotting_context(context=_context), sns.axes_style(style=_style, rc=merged_params):
                func(*args, **kwargs)
        return wrapper

    return decorator


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


def process_tests(dir):
    hz2f_fwd, hz2f_rev = process_calibration(calibration_csv)
    test_csvs = glob.glob(dir + "*.csv")
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


def plot_test_label(*args: str):
    d_label = {'block': 'Block', 'sine': 'Sine', 'r': 'reverse', 'f': 'forward'}
    d_marker = {'block': '^', 'sine': 's'}
    d_color = {('block', 'r'): 'k', ('block', 'f'): 'm', ('sine', 'r'): 'c', ('sine', 'f'): 'r'}
    # label = f'{d_label[args[0]]} {d_label[args[1]]} {args[2]}%'
    label = f'{d_label[args[0]]} {args[2]}%'
    marker = f'{d_marker[args[0]]}'
    color = f'{d_color[tuple([args[0], args[1]])]}'
    return label, marker, color


def plot_tests(data_dict: dict, test_range: str, error_bar: float):
    title = f'Chirp test for thrust {test_range}%'
    plt.figure(title)
    # nfs = 11
    nfs = 16
    for tname, tdata in data_dict.items():
        if test_range in tname:
            chirp_freq = list(range(nfs))
            label, marker_type, color = plot_test_label(*tname.split('_'))
            marker_size = 15
            if tdata.shape[1] == 2:
                try:
                    plt.errorbar(chirp_freq, tdata[:, 0], yerr=error_bar, label=label, color=color, ls='--',
                                 marker=marker_type, markevery=1, ms=marker_size)
                    plt.errorbar(chirp_freq, tdata[:, 1], yerr=error_bar, color=color, ls='--',
                                 marker=marker_type, markevery=1, ms=marker_size)
                except ValueError:
                    chirp_freq = list(range(nfs - 1))
                    plt.errorbar(chirp_freq, tdata[:, 0], yerr=error_bar, label=label, color=color, ls='--',
                                 marker=marker_type, markevery=1, ms=marker_size)
                    plt.errorbar(chirp_freq, tdata[:, 1], yerr=error_bar, color=color, ls='--',
                                 marker=marker_type, markevery=1, ms=marker_size)
            else:
                plt.errorbar(chirp_freq, tdata, yerr=error_bar, label=label, color=color, ls='--',
                             marker=marker_type, markevery=1, ms=marker_size)
    plt.legend()
    plt.title(title)
    plt.ylabel('Thrust [%]')
    plt.xlabel('Chirp frequency [Hz]')
    plt.grid()
    fig = plt.gcf()
    # fig.set_size_inches((15, 11), forward=False)
    fig.tight_layout()
    # fig.savefig(f'./savefigs/{title}.png', bbox_inches='tight')


def my_normalize(array: np.ndarray):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def make_df(thr_dict):
    labels = []
    data = []
    for tname, tdata in thr_dict.items():
        delta_thr = tdata[:, 1] - tdata[:, 0]
        if delta_thr.size == 16:
            delta_thr = delta_thr[:-1]
        label, _, _ = plot_test_label(*tname.split("_"))
        labels.append(label)
        data.append(delta_thr)
    # data = np.vstack(data).T
    d = {}
    for name, arr in zip(labels, data):
        d[name] = pandas.Series(arr, index=list(range(arr.size)))
    df = pandas.DataFrame(d)
    df.rename_axis(columns="names", inplace=True)
    # print(df)
    # quit()
    return df


@prettify_plot()
def plot_bode(thr_dict: dict):
    plt.figure()
    # df = make_df(thr_dict)
    # sns.lineplot(x=df.index, data=df.values.T)
    lines = ["-", "--", "-.", ":"]
    cycle_lines = cycle(lines)
    for tname, tdata in thr_dict.items():
        if "f" in tname:
            delta_thr = tdata[:, 1] - tdata[:, 0]
            if delta_thr.size == 16:
                delta_thr = delta_thr[:-1]
            label, _, _ = plot_test_label(*tname.split("_"))
            plt.plot(list(range(15)), delta_thr, label=label, ls=next(cycle_lines))
    plt.ylabel('Thrust [%]')
    plt.xlabel('Chirp frequency [Hz]')
    plt.legend()


if __name__ == "__main__":
    # calibration_csv = "./calibration.csv"
    # freqs, thrs = process_tests("./test_csv/")

    calibration_csv = "./calibration_31-5.csv"
    freqs, thrs = process_tests("./test_csv_31-5/")

    plt.rcParams.update({'font.size': 18})
    plot_tests(thrs, '10-100', 4.5)
    # plot_tests(thrs, '10-50', 4.5)
    # plot_tests(thrs, '50-100', 4.5)
    plot_bode(thrs, context="poster")
    plt.show()
